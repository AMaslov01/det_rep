"""Strict and legacy DSPy adapters for explicitly configured runtimes.

The research runtime uses native OpenAI ``response_format.type=json_schema``
and deliberately bypasses DSPy's JSON repair and fallback requests.  The old
vLLM 0.6.3 ``guided_json`` adapter remains only for reproducing legacy
artifacts; it is not selected by the new DataSphere profile.  Both adapters
reuse DSPy's typed output schema without changing any KGGen prompt, graph
extraction, or clustering logic.

Imports deliberately stay inside the factory so ordinary offline tests and
remote-provider users do not acquire a DSPy/vLLM dependency.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from functools import wraps
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin


STRUCTURED_OUTPUT_PROTOCOL_VERSION = "strict-response-format-v5-provider-neutral-json-schema"
STRUCTURED_OUTPUT_TRANSPORTS = frozenset({"none", "response_format", "guided_json"})
STRUCTURED_OUTPUT_BACKENDS = frozenset({"xgrammar", "guidance", "vertex"})
XGRAMMAR_STRICT_REQUEST_BACKEND = (
    "xgrammar:disable-any-whitespace,no-fallback"
)


class StructuredOutputError(RuntimeError):
    """Base class for deterministic structured-output contract failures."""


class StructuredOutputSchemaError(StructuredOutputError):
    """The requested or returned document does not satisfy its JSON Schema."""


class StructuredOutputParseError(StructuredOutputError):
    """The server returned something other than one strict JSON document."""


class StructuredOutputTruncatedError(StructuredOutputParseError):
    """The provider stopped a strict response at its output-token limit."""


class GatewayIdentityDriftError(RuntimeError):
    """A completion was served by a gateway other than the pinned manifest."""


def _response_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def validate_gateway_identity(response: Any, *, label: str = "completion") -> None:
    """Check the gateway fingerprint when a runner pins a manifest hash.

    ``EXPECTED_GATEWAY_MANIFEST_SHA256`` is deliberately optional for local
    offline tests and legacy artifacts.  Live controlled runs set it before
    the first request.  The gateway returns ``<manifest-hash-prefix>:<model
    revision>`` as the OpenAI-compatible ``system_fingerprint`` field.
    """
    expected = os.environ.get("EXPECTED_GATEWAY_MANIFEST_SHA256", "").strip().lower()
    if not expected:
        return
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise GatewayIdentityDriftError("EXPECTED_GATEWAY_MANIFEST_SHA256 is malformed")
    actual = _response_field(response, "system_fingerprint")
    if not isinstance(actual, str) or actual.split(":", 1)[0].lower() != expected[:16]:
        raise GatewayIdentityDriftError(
            f"{label} gateway fingerprint differs from the manifest pinned before the run"
        )


def validate_completion_envelope(response: Any, *, label: str = "DSPy completion") -> None:
    """Require one complete choice before DSPy discards transport metadata."""
    validate_gateway_identity(response, label=label)
    choices = _response_field(response, "choices")
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        raise StructuredOutputParseError(f"{label} must contain exactly one choice")
    finish_reason = _response_field(choices[0], "finish_reason")
    if finish_reason == "length":
        raise StructuredOutputTruncatedError(
            f"{label} did not finish cleanly: {finish_reason!r}"
        )
    if finish_reason != "stop":
        raise StructuredOutputParseError(
            f"{label} did not finish cleanly: {finish_reason!r}"
        )


def install_dspy_completion_guard(lm: Any) -> None:
    """Install a version-tolerant finish-reason guard on a DSPy LM instance.

    DSPy 2.6 processes LiteLLM choices into plain strings before an Adapter
    sees them.  Wrapping ``forward`` is the last stable boundary that still
    contains ``choices`` and ``finish_reason``.  The marker makes repeated
    KGExtractor initialisation idempotent.
    """
    if getattr(lm, "_hallu_completion_guard_installed", False):
        return
    original_forward = getattr(lm, "forward", None)
    if not callable(original_forward):
        raise RuntimeError("DSPy LM has no callable forward method to guard")

    @wraps(original_forward)
    def guarded_forward(*args: Any, **kwargs: Any) -> Any:
        response = original_forward(*args, **kwargs)
        validate_completion_envelope(response)
        return response

    lm.forward = guarded_forward
    lm._hallu_completion_guard_installed = True


def strict_json_loads(document: str, *, label: str = "structured output") -> Any:
    """Decode one RFC-8259-style JSON document without permissive extensions.

    Python's standard decoder normally accepts duplicate object keys and the
    non-standard numeric constants ``NaN``/``Infinity``.  Both are ambiguous
    inputs for a scientific structured-output contract: duplicate keys are
    silently resolved by keeping the last value, while non-finite numbers are
    not JSON.  Reject them before independent JSON-Schema validation.
    """
    if not isinstance(document, str):
        raise StructuredOutputParseError(f"{label} is not text")

    def reject_constant(value: str) -> Any:
        raise StructuredOutputParseError(
            f"{label} contains non-JSON numeric constant {value!r}"
        )

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise StructuredOutputParseError(
                    f"{label} contains duplicate object key {key!r}"
                )
            decoded[key] = value
        return decoded

    try:
        return json.loads(
            document,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except StructuredOutputParseError:
        raise
    except json.JSONDecodeError as exc:
        raise StructuredOutputParseError(
            f"{label} is invalid JSON at character {exc.pos}"
        ) from exc


@dataclass(frozen=True)
class StructuredOutputSettings:
    """Validated local structured-output settings derived from ``cfg.llm``."""

    transport: str = "none"
    backend: str = "xgrammar"
    request_backend: str | None = None
    model_revision: str | None = None
    runtime_fingerprint: str | None = None


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def structured_output_settings(llm_config: Any) -> StructuredOutputSettings:
    """Read and validate the explicit structured-output runtime contract.

    ``vllm_guided_json: true`` is retained only so old cached/offline configs
    fail gracefully while Jobs migrate.  It maps to the legacy transport and
    emits a deprecation warning; it never changes ``llm.model``.
    """
    configured_transport = _config_value(llm_config, "structured_output_transport", None)
    legacy_guided = bool(_config_value(llm_config, "vllm_guided_json", False))
    if configured_transport is None:
        transport = "guided_json" if legacy_guided else "none"
        if legacy_guided:
            warnings.warn(
                "llm.vllm_guided_json is deprecated; set "
                "llm.structured_output_transport explicitly",
                DeprecationWarning,
                stacklevel=2,
            )
    else:
        transport = str(configured_transport).strip().lower()
        if legacy_guided and transport != "guided_json":
            raise ValueError(
                "llm.vllm_guided_json conflicts with llm.structured_output_transport"
            )
    if transport not in STRUCTURED_OUTPUT_TRANSPORTS:
        choices = ", ".join(sorted(STRUCTURED_OUTPUT_TRANSPORTS))
        raise ValueError(f"llm.structured_output_transport must be one of: {choices}")

    backend = str(_config_value(llm_config, "structured_output_backend", "xgrammar")).strip().lower()
    if backend not in STRUCTURED_OUTPUT_BACKENDS:
        choices = ", ".join(sorted(STRUCTURED_OUTPUT_BACKENDS))
        raise ValueError(f"llm.structured_output_backend must be one of: {choices}")

    request_backend = _config_value(
        llm_config, "structured_output_request_backend", None
    )
    expected_request_backend = (
        XGRAMMAR_STRICT_REQUEST_BACKEND
        if transport == "response_format" and backend == "xgrammar"
        else None
    )
    if request_backend is not None:
        request_backend = str(request_backend).strip().lower()
        if request_backend != expected_request_backend:
            raise ValueError(
                "llm.structured_output_request_backend must be "
                f"{expected_request_backend!r} for transport={transport!r} "
                f"and backend={backend!r}"
            )
    else:
        request_backend = expected_request_backend

    model_revision = _config_value(llm_config, "model_revision", None)
    runtime_fingerprint = _config_value(llm_config, "runtime_fingerprint", None)
    if transport == "response_format" and (not model_revision or not runtime_fingerprint):
        raise ValueError(
            "native response_format requires exact llm.model_revision and "
            "llm.runtime_fingerprint"
        )
    return StructuredOutputSettings(
        transport=transport,
        backend=backend,
        request_backend=request_backend,
        model_revision=str(model_revision) if model_revision else None,
        runtime_fingerprint=str(runtime_fingerprint) if runtime_fingerprint else None,
    )


def json_schema_response_format(
    schema: Mapping[str, Any], *, name: str = "dspy_program_outputs"
) -> dict[str, Any]:
    """Build the native OpenAI/vLLM ``response_format`` wire object."""
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_") or "structured_output"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": safe_name[:64],
            "schema": deepcopy(dict(schema)),
            "strict": True,
        },
    }


def validate_json_document(instance: Any, schema: Mapping[str, Any]) -> None:
    """Independently validate a decoded response against the exact schema."""
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError, ValidationError
    except Exception as exc:  # pragma: no cover - live dependency check catches this
        raise RuntimeError("jsonschema is required for strict structured outputs") from exc
    try:
        Draft202012Validator.check_schema(dict(schema))
        Draft202012Validator(dict(schema)).validate(instance)
    except SchemaError as exc:
        raise StructuredOutputSchemaError(f"invalid structured-output schema: {exc.message}") from exc
    except ValidationError as exc:
        location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
        raise StructuredOutputSchemaError(
            f"structured output violates schema at {location}: {exc.message}"
        ) from exc


def _fresh_dspy_output_schema(signature: Any) -> dict[str, Any]:
    """Build DSPy's closed output schema without consulting per-call overrides."""
    from dspy.adapters.json_adapter import _get_structured_outputs_response_format

    output_model = _get_structured_outputs_response_format(signature)
    schema = output_model.model_json_schema()
    if not isinstance(schema, Mapping):
        raise StructuredOutputSchemaError("DSPy output model did not produce a JSON Schema object")
    copied = deepcopy(dict(schema))
    # Validate at request construction time so a bad dynamic Literal schema
    # never reaches a paid GPU request.
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(copied)
    except Exception as exc:
        raise StructuredOutputSchemaError("DSPy generated an invalid JSON Schema") from exc
    return copied


def dspy_output_schema(signature: Any) -> dict[str, Any]:
    """Return the exact closed schema shared by request and response parsing.

    Runtime-specialized KGGen signatures carry their own immutable snapshot.
    It lives on a fresh Signature class, not on the adapter, so concurrent and
    async calls with different candidate sets cannot leak schemas into one
    another. Every caller receives a deep copy.
    """
    override = getattr(signature, "_hallu_runtime_output_schema", None)
    if override is not None:
        if not isinstance(override, Mapping):
            raise StructuredOutputSchemaError(
                "runtime structured-output schema override is not an object"
            )
        copied = deepcopy(dict(override))
        try:
            from jsonschema import Draft202012Validator

            Draft202012Validator.check_schema(copied)
        except Exception as exc:
            raise StructuredOutputSchemaError(
                "runtime structured-output schema override is invalid"
            ) from exc
        return copied
    return _fresh_dspy_output_schema(signature)


def _runtime_literal(values: Any) -> Any | None:
    """Return a deterministic ``Literal`` over non-empty runtime strings."""
    if values is None or isinstance(values, (str, bytes)):
        return None
    try:
        normalized = tuple(sorted({str(value) for value in values}))
    except TypeError:
        return None
    if not normalized:
        return None
    return Literal[normalized]


def _specialized_relation_list(annotation: Any, entities: Any) -> Any | None:
    """Bind KGGen relation endpoints to the entities supplied for this call.

    KGGen 0.4's prompt says that subject and object *must* come from its
    ``entities`` input, but the released Pydantic model types both fields as
    unconstrained strings (the upstream source even carries a TODO to use
    Literals).  A constrained decoder therefore cannot enforce the stated
    contract and a small local model can invent a surface alias which the
    official clustering stage cannot map consistently.  Specialising the two
    fields makes the wire schema match KGGen's existing instruction; it does
    not repair or rewrite a model response.
    """
    if get_origin(annotation) is not list:
        return None
    model_args = get_args(annotation)
    if len(model_args) != 1:
        return None
    relation_model = model_args[0]
    endpoint_literal = _runtime_literal(entities)
    if endpoint_literal is None:
        try:
            if entities is not None and len(entities) == 0:
                # The per-call schema postprocessor below makes the relation
                # array empty-only while retaining KGGen's normal list type.
                return annotation
        except TypeError:
            pass
        return None

    try:
        from pydantic import BaseModel, create_model

        if not isinstance(relation_model, type) or not issubclass(
            relation_model, BaseModel
        ):
            return None
        if not {"subject", "predicate", "object"}.issubset(
            relation_model.model_fields
        ):
            return None
        fields: dict[str, tuple[Any, Any]] = {}
        for name, field_info in relation_model.model_fields.items():
            field_type = (
                endpoint_literal
                if name in {"subject", "object"}
                else field_info.annotation
            )
            fields[name] = (field_type, deepcopy(field_info))
        specialized_model = create_model(
            f"{relation_model.__name__}RuntimeEntities",
            __module__=relation_model.__module__,
            **fields,
        )
        specialized_model.__doc__ = relation_model.__doc__
        return list[specialized_model]
    except (AttributeError, TypeError, ValueError):
        return None


def _cluster_representatives(clusters: Any) -> tuple[str, ...]:
    if clusters is None or isinstance(clusters, (str, bytes)):
        return ()
    representatives: set[str] = set()
    try:
        for cluster in clusters:
            if isinstance(cluster, Mapping):
                representative = cluster.get("representative")
            else:
                representative = getattr(cluster, "representative", None)
            if representative is not None:
                representatives.add(str(representative))
    except TypeError:
        return ()
    return tuple(sorted(representatives))


def specialize_dspy_signature(signature: Any, inputs: Mapping[str, Any]) -> Any:
    """Close KGGen 0.4's dynamic output contracts over the actual inputs.

    Three released KGGen signatures express a runtime invariant only in prose
    or capture an earlier superset in their type:

    * relation endpoints must be current ``entities``;
    * an extracted/validated cluster may contain only current candidates;
    * a representative must be a member of the validated cluster.

    The batch assignment signature is likewise limited to an existing
    representative or ``None``. Binding these constraints before formatting
    means the prompt, XGrammar request, independent validator and DSPy
    conversion all use one identical schema. Invalid output is made impossible
    at inference time; no parser repair or post-hoc graph edit is introduced.
    Unknown signatures are returned unchanged.
    """
    update = getattr(signature, "with_updated_fields", None)
    output_fields = getattr(signature, "output_fields", None)
    if not callable(update) or not isinstance(output_fields, Mapping):
        return signature

    specialized = signature
    contract_applied = False
    empty_relation_outputs: set[str] = set()
    assignment_item_count: int | None = None

    for output_name in ("relations", "fixed_relations"):
        field = getattr(specialized, "output_fields", {}).get(output_name)
        if field is None or "entities" not in inputs:
            continue
        relation_list = _specialized_relation_list(
            getattr(field, "annotation", None), inputs.get("entities")
        )
        if relation_list is not None:
            specialized = specialized.with_updated_fields(
                output_name, type_=relation_list
            )
            endpoint_literal = _runtime_literal(inputs.get("entities"))
            if endpoint_literal is not None and "entities" in specialized.input_fields:
                specialized = specialized.with_updated_fields(
                    "entities", type_=list[endpoint_literal]
                )
            contract_applied = True
            try:
                if len(inputs.get("entities")) == 0:
                    empty_relation_outputs.add(output_name)
            except TypeError:
                pass

    if "cluster" in getattr(specialized, "output_fields", {}) and "items" in inputs:
        item_literal = _runtime_literal(inputs.get("items"))
        if item_literal is None:
            raise StructuredOutputSchemaError(
                "KGGen ExtractCluster received no current candidate items"
            )
        specialized = specialized.with_updated_fields(
            "cluster", type_=list[item_literal]
        )
        specialized = specialized.with_updated_fields(
            "items", type_=set[item_literal]
        )
        contract_applied = True

    if (
        "validated_items" in getattr(specialized, "output_fields", {})
        and "cluster" in inputs
    ):
        cluster_literal = _runtime_literal(inputs.get("cluster"))
        if cluster_literal is None:
            raise StructuredOutputSchemaError(
                "KGGen ValidateCluster received an empty candidate cluster"
            )
        specialized = specialized.with_updated_fields(
            "validated_items", type_=list[cluster_literal]
        )
        specialized = specialized.with_updated_fields(
            "cluster", type_=set[cluster_literal]
        )
        contract_applied = True

    if (
        "representative" in getattr(specialized, "output_fields", {})
        and "cluster" in inputs
    ):
        cluster_literal = _runtime_literal(inputs.get("cluster"))
        if cluster_literal is None:
            raise StructuredOutputSchemaError(
                "KGGen ChooseRepresentative received an empty cluster"
            )
        specialized = specialized.with_updated_fields(
            "representative", type_=cluster_literal
        )
        specialized = specialized.with_updated_fields(
            "cluster", type_=set[cluster_literal]
        )
        contract_applied = True

    assignment_name = "cluster_reps_that_items_belong_to"
    if (
        assignment_name in getattr(specialized, "output_fields", {})
        and "items" in inputs
        and "clusters" in inputs
    ):
        representatives = _cluster_representatives(inputs.get("clusters"))
        representative_literal = _runtime_literal(representatives)
        try:
            item_count = len(inputs.get("items"))
        except TypeError:
            item_count = 0
        if representative_literal is None or item_count <= 0:
            raise StructuredOutputSchemaError(
                "KGGen CheckExistingClusters requires items and existing representatives"
            )
        assignment_item = representative_literal | None
        assignments = list[assignment_item]
        specialized = specialized.with_updated_fields(
            assignment_name, type_=assignments
        )
        item_literal = _runtime_literal(inputs.get("items"))
        if item_literal is not None and "items" in specialized.input_fields:
            specialized = specialized.with_updated_fields(
                "items", type_=list[item_literal]
            )
        contract_applied = True
        assignment_item_count = item_count

    if contract_applied:
        runtime_schema = _fresh_dspy_output_schema(specialized)
        properties = runtime_schema.get("properties")
        if not isinstance(properties, dict):
            raise StructuredOutputSchemaError(
                "DSPy runtime output schema has no properties object"
            )
        for output_name in empty_relation_outputs:
            relation_array = properties.get(output_name)
            if not isinstance(relation_array, dict) or relation_array.get("type") != "array":
                raise StructuredOutputSchemaError(
                    f"KGGen {output_name} output is not an array schema"
                )
            relation_array["minItems"] = 0
            relation_array["maxItems"] = 0
        if assignment_item_count is not None and assignment_name in properties:
            assignment_array = properties[assignment_name]
            if not isinstance(assignment_array, dict) or assignment_array.get("type") != "array":
                raise StructuredOutputSchemaError(
                    "KGGen existing-cluster assignments are not an array schema"
                )
            assignment_array["minItems"] = assignment_item_count
            assignment_array["maxItems"] = assignment_item_count
        try:
            from jsonschema import Draft202012Validator

            Draft202012Validator.check_schema(runtime_schema)
        except Exception as exc:
            raise StructuredOutputSchemaError(
                "runtime-specialized DSPy schema is invalid"
            ) from exc
        setattr(
            specialized,
            "_hallu_runtime_output_schema",
            deepcopy(runtime_schema),
        )

    return specialized


def inline_local_json_schema_refs(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Expand local ``#/$defs/...`` references in a Pydantic JSON Schema.

    DSPy serialises ``list[Relation]`` as a perfectly valid Pydantic schema
    whose array item is ``{"$ref": "#/$defs/Relation"}``.  The particular
    vLLM 0.6.3 structured-output backends available in DataSphere accept a
    small schema and start a request for this one, but do not honour that
    local reference: they can emit a bare relation object instead of the
    required ``{"relations": [...]}`` root.  Expanding a local definition at
    its reference site preserves the accepted JSON instances; it merely gives
    the constrained decoder an equivalent schema that does not rely on
    JSON-Pointer resolution.

    Only references into the root ``$defs`` are expanded.  Foreign and cyclic
    references are left intact, together with ``$defs``, rather than risking a
    silent semantic change.  Pydantic's KGGen relation schemas are acyclic and
    use exactly this local-reference form.
    """
    root = deepcopy(dict(schema))
    definitions = root.get("$defs")
    if definitions is None:
        return root
    if not isinstance(definitions, Mapping):
        raise TypeError("JSON Schema $defs must be an object")

    def resolve_local_ref(reference: str) -> Any | None:
        prefix = "#/$defs/"
        if not reference.startswith(prefix):
            return None
        current: Any = definitions
        # JSON Pointer escaping is required for correctness even though
        # Pydantic's current Relation name has no escaped characters.
        for component in reference[len(prefix):].split("/"):
            component = component.replace("~1", "/").replace("~0", "~")
            if not isinstance(current, Mapping) or component not in current:
                raise ValueError(f"unresolvable local JSON Schema reference: {reference}")
            current = current[component]
        return current

    def expand(node: Any, active_refs: frozenset[str] = frozenset()) -> Any:
        if isinstance(node, list):
            return [expand(item, active_refs) for item in node]
        if not isinstance(node, Mapping):
            return deepcopy(node)

        reference = node.get("$ref")
        if isinstance(reference, str):
            target = resolve_local_ref(reference)
            if target is not None:
                # Recursive models are not part of KGGen's output language.
                # Retain their original reference rather than expanding it
                # infinitely or weakening the original schema.
                if reference in active_refs:
                    return {key: expand(value, active_refs) for key, value in node.items()}
                expanded_target = expand(target, active_refs | {reference})
                siblings = {
                    key: expand(value, active_refs)
                    for key, value in node.items()
                    if key != "$ref"
                }
                if not siblings:
                    return expanded_target
                # JSON Schema permits sibling constraints next to ``$ref``.
                # ``allOf`` has the same intersection semantics without
                # overwriting either the target or those constraints.
                return {"allOf": [expanded_target, siblings]}

        return {key: expand(value, active_refs) for key, value in node.items()}

    expanded = expand(root)

    def contains_local_def_ref(node: Any) -> bool:
        if isinstance(node, Mapping):
            if isinstance(node.get("$ref"), str) and node["$ref"].startswith("#/$defs/"):
                return True
            return any(contains_local_def_ref(value) for value in node.values())
        if isinstance(node, list):
            return any(contains_local_def_ref(value) for value in node)
        return False

    # Do not remove definitions if a recursive reference could not be safely
    # inlined.  For KGGen's acyclic Relation schema this removes ``$defs`` and
    # produces the exact flat grammar required by vLLM 0.6.3.
    if not contains_local_def_ref(expanded):
        expanded.pop("$defs", None)
    return expanded


_NON_SEMANTIC_SCHEMA_ANNOTATIONS = frozenset({
    "title",
    "description",
    "examples",
    "default",
    "$comment",
    # ``desc`` is DSPy's legacy field metadata, not a JSON Schema keyword.
    "desc",
})
_SCHEMA_MAP_KEYWORDS = frozenset({"$defs", "patternProperties", "dependentSchemas"})


def canonicalize_vllm_guided_json_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the constraint-bearing JSON Schema accepted by vLLM 0.6.3.

    In addition to local ``$defs`` expansion, remove annotations and DSPy's
    private ``__dspy_*`` metadata.  These keywords do not restrict which JSON
    documents validate; they are prompt/UI metadata.  Keeping them in the
    grammar sent to the old Outlines backend makes it silently ignore the
    enclosing object for KGGen's dynamic fallback Relation schema.  Removing
    them therefore preserves exactly the same JSON language while avoiding
    that backend defect.
    """
    expanded = inline_local_json_schema_refs(schema)

    def clean(node: Any, *, property_map: bool = False) -> Any:
        if isinstance(node, list):
            return [clean(value) for value in node]
        if not isinstance(node, Mapping):
            return deepcopy(node)
        if property_map:
            # A user is allowed to name a JSON property ``title`` or ``desc``;
            # these are property names here, not schema annotations.
            return {str(key): clean(value) for key, value in node.items()}

        result: dict[str, Any] = {}
        for key, value in node.items():
            if key in _NON_SEMANTIC_SCHEMA_ANNOTATIONS or key.startswith("__dspy_"):
                continue
            if key == "properties":
                if not isinstance(value, Mapping):
                    raise TypeError("JSON Schema properties must be an object")
                result[key] = clean(value, property_map=True)
            elif key in _SCHEMA_MAP_KEYWORDS:
                if not isinstance(value, Mapping):
                    raise TypeError(f"JSON Schema {key} must be an object")
                result[key] = clean(value, property_map=True)
            else:
                result[key] = clean(value)
        return result

    return clean(expanded)


def strict_json_schema_adapter(*, request_backend: str | None = None) -> Any:
    """Return a DSPy JSON adapter with one strict native-schema request.

    DSPy 2.6's stock :class:`JSONAdapter` catches *any* structured-output
    exception and silently sends a second ``json_object`` request.  Its parser
    also uses ``json_repair``.  Both behaviours are useful for casual chat,
    but invalidate an extraction experiment because a transport failure can
    be hidden or a truncated response can be changed locally.  This adapter:

    * always sends the exact DSPy-generated closed schema through native
      ``response_format``;
    * performs exactly one LM call;
    * parses with ``json.loads`` only; and
    * independently validates the raw document before DSPy type conversion.
    """
    from dspy.adapters.base import Adapter
    from dspy.adapters.json_adapter import JSONAdapter
    from dspy.adapters.utils import parse_value

    class StrictJSONSchemaAdapter(JSONAdapter):
        @staticmethod
        def _singleton_representative(
            signature: Any, inputs: Mapping[str, Any]
        ) -> list[dict[str, str]] | None:
            """Return the only schema-valid representative without an LLM call.

            KGGen's clustering loop asks ``choose_rep`` even for a validated
            one-item cluster.  In that case the output contract has exactly one
            possible value, so invoking a model cannot add information and a
            provider's schema violation only turns a no-op into a failed graph.
            This is not response repair: no model document is accepted, and the
            returned value is the sole member required by KGGen's own contract.
            """
            output_fields = getattr(signature, "output_fields", {})
            if set(output_fields) != {"representative"} or "cluster" not in inputs:
                return None
            cluster = inputs["cluster"]
            if isinstance(cluster, (str, bytes)):
                return None
            try:
                members = sorted({str(member) for member in cluster})
            except TypeError:
                return None
            if len(members) != 1:
                return None
            return [{"representative": members[0]}]

        @staticmethod
        def _request_kwargs(
            lm_kwargs: Mapping[str, Any], signature: Any
        ) -> dict[str, Any]:
            """Return an isolated request containing only the native schema control.

            Calling the DSPy base ``Adapter`` directly below is intentional. In the
            pinned DSPy 2.6.27 implementation, ``JSONAdapter.__call__`` catches
            every structured-output failure and issues a second ``json_object``
            request.  ``ChatAdapter.__call__`` also owns a fallback.  The base
            adapter owns only format -> one LM call -> parse, which is the
            research contract we need and is stable across DSPy 2.6 and 3.x.
            """
            schema = dspy_output_schema(signature)
            request_kwargs = dict(lm_kwargs)
            previous_extra = request_kwargs.get("extra_body")
            if previous_extra is not None:
                if not isinstance(previous_extra, Mapping):
                    raise TypeError("DSPy lm_kwargs.extra_body must be a mapping")
                cleaned_extra = dict(previous_extra)
                cleaned_extra.pop("guided_json", None)
                if cleaned_extra:
                    request_kwargs["extra_body"] = cleaned_extra
                else:
                    request_kwargs.pop("extra_body", None)
            if request_backend is not None:
                previous_extra = request_kwargs.get("extra_body", {})
                if not isinstance(previous_extra, Mapping):
                    raise TypeError("DSPy lm_kwargs.extra_body must be a mapping")
                request_kwargs["extra_body"] = {
                    **dict(previous_extra),
                    "guided_decoding_backend": request_backend,
                }
            request_kwargs["response_format"] = json_schema_response_format(
                schema,
                name=getattr(signature, "__name__", "dspy_program_outputs"),
            )
            return request_kwargs

        def parse(self, signature: Any, completion: str) -> dict[str, Any]:
            fields = strict_json_loads(completion, label="DSPy completion")
            if not isinstance(fields, dict):
                raise StructuredOutputParseError("structured output root must be an object")

            expected = set(signature.output_fields)
            actual = set(fields)
            if actual != expected:
                raise StructuredOutputSchemaError(
                    "structured output keys differ from the DSPy signature: "
                    f"expected={sorted(expected)!r} actual={sorted(actual)!r}"
                )
            schema = dspy_output_schema(signature)
            validate_json_document(fields, schema)

            converted: dict[str, Any] = {}
            for key, value in fields.items():
                try:
                    converted[key] = parse_value(value, signature.output_fields[key].annotation)
                except Exception as exc:
                    raise StructuredOutputParseError(
                        f"DSPy could not convert structured field {key!r}"
                    ) from exc
            return converted

        def __call__(
            self,
            lm: Any,
            lm_kwargs: dict[str, Any],
            signature: Any,
            demos: list[dict[str, Any]],
            inputs: dict[str, Any],
        ) -> list[dict[str, Any]]:
            singleton = self._singleton_representative(signature, inputs)
            if singleton is not None:
                return singleton
            contract_signature = specialize_dspy_signature(signature, inputs)
            try:
                return Adapter.__call__(
                    self,
                    lm,
                    self._request_kwargs(lm_kwargs, contract_signature),
                    contract_signature,
                    demos,
                    inputs,
                )
            except Exception as exc:
                # Deliberately expose only the request's structural role and
                # exception class: inputs may contain source text or graph
                # values and must never enter progress/log diagnostics.
                print(
                    "[structured-output] request_failed "
                    f"signature={getattr(contract_signature, '__name__', 'unknown')} "
                    f"exception_type={type(exc).__name__}",
                    flush=True,
                )
                raise

        async def acall(
            self,
            lm: Any,
            lm_kwargs: dict[str, Any],
            signature: Any,
            demos: list[dict[str, Any]],
            inputs: dict[str, Any],
        ) -> list[dict[str, Any]]:
            # Do not let an async DSPy caller silently bypass the schema or
            # re-enter JSONAdapter's fallback path.
            singleton = self._singleton_representative(signature, inputs)
            if singleton is not None:
                return singleton
            contract_signature = specialize_dspy_signature(signature, inputs)
            return await Adapter.acall(
                self,
                lm,
                self._request_kwargs(lm_kwargs, contract_signature),
                contract_signature,
                demos,
                inputs,
            )

    return StrictJSONSchemaAdapter()


def is_retryable_llm_exception(exc: BaseException) -> bool:
    """Return true only for transient transport/server failures.

    The check is dependency-light so it works with LiteLLM, httpx, urllib and
    test doubles without importing the live inference stack.  Structured
    schema/parse/Pydantic errors are deterministic and therefore fail fast.
    """
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__

    deterministic_names = {
        "AdapterParseError",
        "JSONDecodeError",
        "SchemaError",
        "ValidationError",
        "RateLimitRetryDeadlineExceeded",
        "RetryDeadlineExceeded",
    }
    retry_statuses = {408, 409, 425, 429}
    transient_names = {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
        # httpx/urllib/aiohttp and gateway wrappers use these names while
        # often discarding the underlying socket exception and HTTP status.
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectError",
        "RemoteProtocolError",
        "ProtocolError",
        "NetworkError",
        "NameResolutionError",
        "ClientConnectorError",
        "ClientOSError",
    }

    # Inspect the exception chain from the current failure outwards.  A
    # transient exception raised while handling an older parse/limit error
    # receives that older error as ``__context__`` in Python.  Looking for a
    # deterministic exception anywhere in the chain would therefore turn a
    # fresh 429 into a hard failure.  The first recognizable cause wins:
    # deterministic structured output stays fail-fast, while a newer
    # transport/provider failure retains its retry/fallback policy.
    for item in chain:
        if (
            isinstance(item, StructuredOutputError)
            or type(item).__name__ in deterministic_names
        ):
            return False
        response = getattr(item, "response", None)
        status = getattr(item, "status_code", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        if status is not None:
            try:
                numeric_status = int(status)
                return numeric_status in retry_statuses or 500 <= numeric_status < 600
            except (TypeError, ValueError):
                return False
        if isinstance(item, (TimeoutError, ConnectionError)):
            return True
        if type(item).__name__ in transient_names:
            return True

    # Some LiteLLM/OpenAI compatibility layers discard ``response`` and
    # ``status_code`` while wrapping an otherwise ordinary Exception.  Do not
    # abandon a long, cache-backed Job merely because that wrapper lost its
    # typed HTTP metadata: recognise only explicit transient HTTP/capacity
    # signatures, after deterministic structured-output failures were already
    # excluded above.
    transient_message = " ".join(str(item) for item in chain).lower()
    if re.search(r"(?<!\d)(?:408|409|425|429|5\d\d)(?!\d)", transient_message):
        return True
    return any(
        phrase in transient_message
        for phrase in (
            "rate limit",
            "rate_limit",
            "capacity is temporarily exhausted",
            "service unavailable",
            "connection reset",
            "connection aborted",
            "connection refused",
            "connection timed out",
            "timed out",
            "read timeout",
            "connect timeout",
            "temporarily unavailable",
            "temporary failure in name resolution",
            "name or service not known",
            "dns lookup",
            "http2 framing",
            "unexpected eof",
        )
    )


def vllm_guided_json_adapter() -> Any:
    """Return a DSPy adapter that sends each output schema as ``guided_json``.

    ``JSONAdapter`` already owns DSPy's JSON prompt and parser behaviour.  We
    reuse its private schema builder (pinned together with DSPy in the
    DataSphere runtime) and call ``ChatAdapter`` directly only to avoid
    replacing the vLLM-native constrained parameter with the broken
    ``response_format`` route.  A schema-building failure intentionally falls
    back to DSPy's normal ``JSONAdapter`` instead of silently weakening output
    validation.
    """
    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.adapters.json_adapter import JSONAdapter, _get_structured_outputs_response_format

    class VLLMGuidedJSONAdapter(JSONAdapter):
        def __call__(
            self,
            lm: Any,
            lm_kwargs: dict[str, Any],
            signature: Any,
            demos: list[dict[str, Any]],
            inputs: dict[str, Any],
        ) -> list[dict[str, Any]]:
            try:
                output_model = _get_structured_outputs_response_format(signature)
                schema = canonicalize_vllm_guided_json_schema(output_model.model_json_schema())
            except Exception:
                # Preserve DSPy's own compatibility fallback for signatures
                # which cannot be represented as a closed JSON schema.
                return super().__call__(lm, lm_kwargs, signature, demos, inputs)

            previous_extra = lm_kwargs.get("extra_body", {})
            if previous_extra is None:
                previous_extra = {}
            if not isinstance(previous_extra, dict):
                raise TypeError("DSPy lm_kwargs.extra_body must be a mapping")
            lm_kwargs["extra_body"] = {**previous_extra, "guided_json": schema}
            # Do not send both controls: vLLM 0.6.3.post1 can ignore
            # response_format even when a guided-decoding backend is enabled.
            lm_kwargs.pop("response_format", None)
            # JSONAdapter supplies the JSON-oriented formatting methods via
            # dynamic dispatch.  ChatAdapter supplies the one-call execution
            # path and, because this object is still a JSONAdapter instance,
            # re-raises a parse error rather than retrying unconstrained text.
            return ChatAdapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)

    return VLLMGuidedJSONAdapter()
