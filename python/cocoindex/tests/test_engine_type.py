import dataclasses
import datetime
import uuid
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
import pytest

from cocoindex.typing import (
    KEY_FIELD_NAME,
    TypeAttr,
    Vector,
    VectorInfo,
)
from cocoindex._internal.datatype import analyze_type_info
from cocoindex.engine_type import (
    BasicValueType,
    EnrichedValueType,
    FieldSchema,
    StructSchema,
    StructType,
    TableType,
    VectorTypeSchema,
    decode_value_type,
    encode_enriched_type,
    encode_enriched_type_info,
    encode_value_type,
    enriched_value_type_from_type,
    enriched_value_type_from_type_info,
)


@dataclasses.dataclass
class SimpleDataclass:
    name: str
    value: int


@dataclasses.dataclass
class SimpleDataclassWithDescription:
    """This is a simple dataclass with a description."""

    name: str
    value: int


class SimpleNamedTuple(NamedTuple):
    name: str
    value: int


@dataclasses.dataclass
class CompositeKey:
    name: str
    value: int


def basic_value_type(kind: Any) -> BasicValueType:
    return BasicValueType(kind=kind)


def basic_field(name: str, kind: Any) -> FieldSchema:
    return FieldSchema(
        name=name, value_type=EnrichedValueType(type=basic_value_type(kind))
    )


def enriched_type_from_annotation(t: Any) -> EnrichedValueType:
    return enriched_value_type_from_type_info(analyze_type_info(t))


def test_encode_enriched_type_none() -> None:
    typ = None
    result = encode_enriched_type(typ)
    assert result is None


def test_enriched_value_type_from_type_none() -> None:
    assert enriched_value_type_from_type(None) is None


def test_enriched_value_type_from_dataclass() -> None:
    typ = SimpleDataclass
    result = enriched_type_from_annotation(typ)
    assert result == EnrichedValueType(
        type=StructType(
            fields=[
                basic_field("name", "Str"),
                basic_field("value", "Int64"),
            ],
            description="SimpleDataclass(name: str, value: int)",
        ),
    )


def test_enriched_value_type_from_dataclass_with_description() -> None:
    typ = SimpleDataclassWithDescription
    result = enriched_type_from_annotation(typ)
    assert result == EnrichedValueType(
        type=StructType(
            fields=[
                basic_field("name", "Str"),
                basic_field("value", "Int64"),
            ],
            description="This is a simple dataclass with a description.",
        ),
    )


def test_enriched_value_type_from_named_tuple() -> None:
    typ = SimpleNamedTuple
    result = enriched_type_from_annotation(typ)
    assert result == EnrichedValueType(
        type=StructType(
            fields=[
                basic_field("name", "Str"),
                basic_field("value", "Int64"),
            ],
            description="SimpleNamedTuple(name, value)",
        ),
    )


def test_enriched_value_type_from_vector() -> None:
    typ = NDArray[np.float32]
    result = enriched_type_from_annotation(typ)
    assert result == EnrichedValueType(
        type=BasicValueType(
            kind="Vector",
            vector=VectorTypeSchema(
                element_type=basic_value_type("Float32"),
                dimension=None,
            ),
        ),
    )


def test_enriched_value_type_from_ltable() -> None:
    typ = list[SimpleDataclass]
    result = enriched_type_from_annotation(typ)
    assert result == EnrichedValueType(
        type=TableType(
            kind="LTable",
            row=StructSchema(
                fields=[
                    basic_field("name", "Str"),
                    basic_field("value", "Int64"),
                ],
                description="SimpleDataclass(name: str, value: int)",
            ),
        ),
    )


def test_enriched_value_type_from_ktable() -> None:
    typ = dict[str, SimpleDataclass]
    result = enriched_type_from_annotation(typ)
    assert result == EnrichedValueType(
        type=TableType(
            kind="KTable",
            row=StructSchema(
                fields=[
                    basic_field(KEY_FIELD_NAME, "Str"),
                    basic_field("name", "Str"),
                    basic_field("value", "Int64"),
                ],
                description="SimpleDataclass(name: str, value: int)",
            ),
            num_key_parts=1,
        ),
    )


def test_enriched_value_type_from_ktable_with_composite_key() -> None:
    typ = dict[CompositeKey, SimpleDataclass]
    result = enriched_type_from_annotation(typ)
    assert result == EnrichedValueType(
        type=TableType(
            kind="KTable",
            row=StructSchema(
                fields=[
                    basic_field("name", "Str"),
                    basic_field("value", "Int64"),
                    basic_field("name", "Str"),
                    basic_field("value", "Int64"),
                ],
                description="SimpleDataclass(name: str, value: int)",
            ),
            num_key_parts=2,
        ),
    )


def test_enriched_value_type_with_attrs() -> None:
    typ = Annotated[str, TypeAttr("key", "value")]
    result = enriched_type_from_annotation(typ)
    assert result == EnrichedValueType(
        type=basic_value_type("Str"),
        attrs={"key": "value"},
    )


def test_enriched_value_type_nullable() -> None:
    typ = str | None
    result = enriched_type_from_annotation(typ)
    assert result == EnrichedValueType(
        type=basic_value_type("Str"),
        nullable=True,
    )


def test_encode_scalar_numpy_types_schema() -> None:
    for np_type, expected_kind in [
        (np.int64, "Int64"),
        (np.float32, "Float32"),
        (np.float64, "Float64"),
    ]:
        schema = enriched_type_from_annotation(np_type)
        assert schema == EnrichedValueType(type=basic_value_type(expected_kind))


def test_encode_enriched_type_encodes_schema() -> None:
    for typ in [SimpleDataclass, dict[str, SimpleDataclass]]:
        schema = enriched_type_from_annotation(typ)
        assert enriched_value_type_from_type(typ) == schema
        assert encode_enriched_type(typ) == schema.encode()
        assert encode_enriched_type_info(analyze_type_info(typ)) == schema.encode()


def test_vector_element_must_be_basic_type() -> None:
    with pytest.raises(
        ValueError, match="Vector element type must be a basic value type"
    ):
        enriched_type_from_annotation(list[list[SimpleDataclass]])


def test_union_variants_must_be_basic_types() -> None:
    with pytest.raises(
        ValueError, match="Union variant type must be a basic value type"
    ):
        enriched_type_from_annotation(SimpleDataclass | int)


# ========================= Encode/Decode Tests =========================


def schema_from_type_info(t: Any) -> EnrichedValueType:
    return enriched_value_type_from_type_info(analyze_type_info(t))


def test_basic_types_encode_decode() -> None:
    """Test encode/decode roundtrip for basic Python types."""
    test_cases = [
        str,
        int,
        float,
        bool,
        bytes,
        uuid.UUID,
        datetime.date,
        datetime.time,
        datetime.datetime,
        datetime.timedelta,
    ]

    for typ in test_cases:
        encoded = schema_from_type_info(typ).encode()
        decoded = decode_value_type(encoded["type"])
        reencoded = encode_value_type(decoded)
        assert reencoded == encoded["type"]


def test_vector_types_encode_decode() -> None:
    """Test encode/decode roundtrip for vector types."""
    test_cases = [
        NDArray[np.float32],
        NDArray[np.float64],
        NDArray[np.int64],
        Vector[np.float32],
        Vector[np.float32, Literal[128]],
        Vector[str],
    ]

    for typ in test_cases:
        encoded = schema_from_type_info(typ).encode()
        decoded = decode_value_type(encoded["type"])
        reencoded = encode_value_type(decoded)
        assert reencoded == encoded["type"]


def test_struct_types_encode_decode() -> None:
    """Test encode/decode roundtrip for struct types."""
    test_cases = [
        SimpleDataclass,
        SimpleNamedTuple,
    ]

    for typ in test_cases:
        encoded = schema_from_type_info(typ).encode()
        decoded = decode_value_type(encoded["type"])
        reencoded = encode_value_type(decoded)
        assert reencoded == encoded["type"]


def test_table_types_encode_decode() -> None:
    """Test encode/decode roundtrip for table types."""
    test_cases = [
        list[SimpleDataclass],  # LTable
        dict[str, SimpleDataclass],  # KTable
    ]

    for typ in test_cases:
        encoded = schema_from_type_info(typ).encode()
        decoded = decode_value_type(encoded["type"])
        reencoded = encode_value_type(decoded)
        assert reencoded == encoded["type"]


def test_nullable_types_encode_decode() -> None:
    """Test encode/decode roundtrip for nullable types."""
    test_cases = [
        str | None,
        int | None,
        NDArray[np.float32] | None,
    ]

    for typ in test_cases:
        encoded = schema_from_type_info(typ).encode()
        decoded = decode_value_type(encoded["type"])
        reencoded = encode_value_type(decoded)
        assert reencoded == encoded["type"]


def test_annotated_types_encode_decode() -> None:
    """Test encode/decode roundtrip for annotated types."""
    test_cases = [
        Annotated[str, TypeAttr("key", "value")],
        Annotated[NDArray[np.float32], VectorInfo(dim=256)],
        Annotated[list[int], VectorInfo(dim=10)],
    ]

    for typ in test_cases:
        encoded = schema_from_type_info(typ).encode()
        decoded = decode_value_type(encoded["type"])
        reencoded = encode_value_type(decoded)
        assert reencoded == encoded["type"]


def test_complex_nested_encode_decode() -> None:
    """Test complex nested structure encode/decode roundtrip."""

    # Create a complex nested structure using Python type annotations
    @dataclasses.dataclass
    class ComplexStruct:
        embedding: NDArray[np.float32]
        metadata: str | None
        score: Annotated[float, TypeAttr("indexed", True)]

    encoded = schema_from_type_info(ComplexStruct).encode()
    decoded = decode_value_type(encoded["type"])
    reencoded = encode_value_type(decoded)
    assert reencoded == encoded["type"]
