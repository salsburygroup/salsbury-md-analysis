"""Shared residue-name vocabulary for conservative generic-system inference.

These names are routing aids, not a substitute for an explicit topology or
force-field chemistry.  Callers still use atom identity and connectivity for
scientific calculations, and unusual residues remain reviewable overrides.
"""

from __future__ import annotations


WATER_RESIDUES = frozenset({
    "HOH", "WAT", "SOL", "TIP", "TIP3", "TIP3P", "TP3", "SPC", "SPCE",
    "OPC", "OPC3", "TIP4", "TIP4P", "TIP5P",
})

PROTEIN_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYM", "CYX", "GLN", "GLU",
    "GLH", "GLY", "HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "ILE",
    "LEU", "LYS", "LYN", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR",
    "VAL", "SEC", "PYL", "ACE", "NME",
})

STANDARD_NUCLEIC_RESIDUES = frozenset({
    "A", "C", "G", "T", "U", "DA", "DC", "DG", "DT", "DU", "RA", "RC",
    "RG", "RU", "ADE", "CYT", "GUA", "THY", "URA",
})

NUCLEIC_RESIDUES = STANDARD_NUCLEIC_RESIDUES | frozenset({
    "8OG", "8OX", "OX3", "EDU", "FDU", "5MC", "PSU",
})

ION_RESIDUES = frozenset({
    "LI", "LI+", "NA", "NA+", "SOD", "K", "K+", "POT", "RB", "CS", "MG",
    "MG2", "MG2+", "CA", "CA2", "CA2+", "SR", "BA", "ZN", "ZN2", "ZN2+",
    "FE", "FE2", "FE3", "CU", "CU1", "CU2", "MN", "MN2", "CO", "CO2",
    "NI", "NI2", "CD", "HG", "CL", "CL-", "CLA", "BR", "BR-", "I",
    "I-", "F", "IOD",
})

CATION_ELEMENTS = frozenset({
    "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "MN", "FE",
    "CO", "NI", "CU", "ZN", "CD", "HG",
})

ANION_ELEMENTS = frozenset({"F", "CL", "BR", "I"})

SOLVENT_AND_ION_RESIDUES = WATER_RESIDUES | ION_RESIDUES
