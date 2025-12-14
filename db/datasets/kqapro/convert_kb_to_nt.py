import json
from rdflib import Graph, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD, Namespace
from pathlib import Path

"""This File converts the original kb.json file into kb.nt, a Fileformat which can be read by rdf based vector databases like Qlever or Virtuoso. """

# --- 1. Setup Namespaces ---
# We create base URIs for our data.
# You can change 'http://example.org/' to your own domain.
BASE = "http://kqapro.org/"
EX = Namespace(BASE + "entity/")      # For entities and concepts
PROP = Namespace(BASE + "property/")    # For relation predicates
ATTR = Namespace(BASE + "attribute/")   # For attribute keys
QUAL = Namespace(BASE + "qualifier/")   # For qualifier keys
UNIT = Namespace(BASE + "unit/")        # For units


def sanitize_for_uri(s: str) -> str:
    """Replaces spaces with underscores for URI components."""
    if s is None:
        return ""  # Return empty string if None
    return s.replace(' ', '_')


def create_literal_or_bnode(g, value_obj):
    """
    Creates an rdflib Literal or BNode based on the value's type.
    Quantities with units are converted to BNodes.
    """
    v_type = value_obj.get('type')
    v_val = value_obj.get('value')

    if v_type == 'string':
        # String literals should preserve spaces
        return Literal(v_val)

    # --- MODIFIED ---
    elif v_type == 'date':
        # XSD.date requires YYYY-MM-DD format. Your JSON uses YYYY/MM/DD.
        # We must replace '/' with '-' AND zero-pad the month and day.
        try:
            parts = str(v_val).split('/')
            if len(parts) == 3:
                year, month, day = parts
                # Format to YYYY-MM-DD with zero-padding
                v_val_formatted = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
                return Literal(v_val_formatted, datatype=XSD.date)
            else:
                # Fallback for unexpected format, just try replacing
                print(f"Warning: Unexpected date format '{v_val}'. Attempting simple replace.")
                v_val_formatted = str(v_val).replace('/', '-')
                return Literal(v_val_formatted, datatype=XSD.date)
        except Exception as e:
            # Handle potential conversion errors (e.g., 'value' is not Y/M/D or is None)
            print(f"Warning: Could not parse date '{v_val}'. Error: {e}. Storing as plain string.")
            return Literal(str(v_val))  # Store as plain string if formatting fails

    elif v_type == 'year':
        try:
            # Format to YYYY
            v_val_formatted = f"{int(v_val):04d}"
            return Literal(v_val_formatted, datatype=XSD.gYear)
        except Exception as e:
            print(f"Warning: Could not parse year '{v_val}'. Error: {e}. Storing as plain string.")
            return Literal(str(v_val))
    # --- END MODIFIED ---

    elif v_type == 'quantity':
        # If it has a unit, we create a BNode to hold value and unit
        v_unit = value_obj.get('unit')
        if v_unit:
            q_node = BNode()  # Create a blank node for the quantity
            g.add((q_node, RDF.value, Literal(v_val, datatype=XSD.decimal)))

            # Treat unit as a URI, not a Literal, and sanitize spaces
            unit_uri = UNIT[sanitize_for_uri(v_unit)]
            g.add((q_node, UNIT.unit, unit_uri))

            return q_node
        else:
            # No unit, just treat it as a number
            return Literal(v_val, datatype=XSD.decimal)

    # Fallback (and ensure value is not None)
    return Literal(str(v_val))


def add_qualifiers(g, statement_node, qualifiers_dict):
    """
    Adds qualifiers to a reified statement node.
    """
    for qk, qv_list in qualifiers_dict.items():
        # --- MODIFIED ---
        # Sanitize qualifier key (predicate) for URI
        qual_prop = QUAL[sanitize_for_uri(qk)]
        # --- END MODIFIED ---

        for qv_obj in qv_list:
            qual_val = create_literal_or_bnode(g, qv_obj)
            g.add((statement_node, qual_prop, qual_val))


def convert_kb_to_rdf(json_file_path, output_nt_file_path):
    print(f"Loading '{json_file_path}'...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    g = Graph()
    # Bind namespaces for cleaner Turtle output (though N-Triples don't use it)
    g.bind("ex", EX)
    g.bind("prop", PROP)
    g.bind("attr", ATTR)
    g.bind("qual", QUAL)
    g.bind("unit", UNIT)
    g.bind("rdfs", RDFS)

    # --- 2. Process Concepts ---
    print("Processing concepts...")
    for concept_id, concept_data in data.get('concepts', {}).items():
        # --- MODIFIED ---
        # Sanitize concept ID for URI
        subject_uri = EX[sanitize_for_uri(concept_id)]
        # --- END MODIFIED ---

        # Add name as rdfs:label (spaces are preserved)
        g.add((subject_uri, RDFS.label, Literal(concept_data.get('name'))))

        # Add instanceOf as rdf:type
        for parent_id in concept_data.get('instanceOf', []):
            # --- MODIFIED ---
            # Sanitize parent ID for URI
            g.add((subject_uri, RDF.type, EX[sanitize_for_uri(parent_id)]))
            # --- END MODIFIED ---

    # --- 3. Process Entities ---
    print("Processing entities...")
    for entity_id, entity_data in data.get('entities', {}).items():
        # --- MODIFIED ---
        # Sanitize entity ID for URI
        subject_uri = EX[sanitize_for_uri(entity_id)]
        # --- END MODIFIED ---

        # Add name as rdfs:label (spaces are preserved)
        g.add((subject_uri, RDFS.label, Literal(entity_data.get('name'))))

        # Add instanceOf as rdf:type
        for parent_id in entity_data.get('instanceOf', []):
            # --- MODIFIED ---
            # Sanitize parent ID for URI
            g.add((subject_uri, RDF.type, EX[sanitize_for_uri(parent_id)]))
            # --- END MODIFIED ---

        # --- 4. Process Attributes (with Qualifiers) ---
        for attr in entity_data.get('attributes', []):
            # --- MODIFIED ---
            # Sanitize attribute key (predicate) for URI
            attr_prop = ATTR[sanitize_for_uri(attr.get('key'))]
            # --- END MODIFIED ---

            attr_value_node = create_literal_or_bnode(g, attr.get('value', {}))

            # Add the simple triple
            g.add((subject_uri, attr_prop, attr_value_node))

            # Handle qualifiers via reification
            qualifiers = attr.get('qualifiers')
            if qualifiers:
                statement_node = BNode()  # Blank node for the statement
                g.add((statement_node, RDF.type, RDF.Statement))
                g.add((statement_node, RDF.subject, subject_uri))  # subject_uri is already sanitized
                g.add((statement_node, RDF.predicate, attr_prop))  # attr_prop is already sanitized
                g.add((statement_node, RDF.object, attr_value_node))
                add_qualifiers(g, statement_node, qualifiers)

        # --- 5. Process Relations (with Qualifiers) ---
        for rel in entity_data.get('relations', []):
            # --- MODIFIED ---
            # Sanitize predicate and object for URIs
            predicate = PROP[sanitize_for_uri(rel.get('predicate'))]
            object_uri = EX[sanitize_for_uri(rel.get('object'))]
            # --- END MODIFIED ---

            # Handle direction
            if rel.get('direction') == 'forward':
                s, o = subject_uri, object_uri  # subject_uri is already sanitized
            elif rel.get('direction') == 'backward':
                s, o = object_uri, subject_uri  # object_uri is now sanitized
            else:
                continue  # Skip if direction is unknown

            # Add the simple triple
            g.add((s, predicate, o))

            # Handle qualifiers via reification
            qualifiers = rel.get('qualifiers')
            if qualifiers:
                statement_node = BNode()  # Blank node for the statement
                g.add((statement_node, RDF.type, RDF.Statement))
                g.add((statement_node, RDF.subject, s))  # s is sanitized
                g.add((statement_node, RDF.predicate, predicate))  # predicate is sanitized
                g.add((statement_node, RDF.object, o))  # o is sanitized
                add_qualifiers(g, statement_node, qualifiers)

    # --- 6. Serialize and Save ---
    print(f"Serializing to '{output_nt_file_path}'...")
    g.serialize(destination=output_nt_file_path, format='nt')
    print("Conversion complete.")


# --- Main execution ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()

    # Define paths relative to this script's directory
    input_file = script_dir / 'kb.json'
    output_file = script_dir / 'kb.nt'

    convert_kb_to_rdf(
        json_file_path=input_file,
        output_nt_file_path=output_file
    )
