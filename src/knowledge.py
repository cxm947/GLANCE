from __future__ import annotations

TACTIC_ORDER = {
    "TA0043": 0,
    "TA0042": 1,
    "TA0001": 2,
    "TA0002": 3,
    "TA0003": 4,
    "TA0004": 5,
    "TA0005": 6,
    "TA0006": 7,
    "TA0007": 8,
    "TA0008": 9,
    "TA0009": 10,
    "TA0011": 11,
    "TA0010": 12,
    "TA0040": 13,
}

TACTIC_NAMES = {
    "TA0043": "Reconnaissance",
    "TA0042": "Resource Development",
    "TA0001": "Initial Access",
    "TA0002": "Execution",
    "TA0003": "Persistence",
    "TA0004": "Privilege Escalation",
    "TA0005": "Defense Evasion",
    "TA0006": "Credential Access",
    "TA0007": "Discovery",
    "TA0008": "Lateral Movement",
    "TA0009": "Collection",
    "TA0011": "Command and Control",
    "TA0010": "Exfiltration",
    "TA0040": "Impact",
}

TACTIC_ID_BY_NAME = {v.lower(): k for k, v in TACTIC_NAMES.items()}

TECHNIQUE_TACTIC_MAP = {
    "T1583": "TA0042", "T1583.001": "TA0042", "T1583.006": "TA0042",
    "T1566": "TA0001", "T1566.001": "TA0001", "T1566.002": "TA0001",
    "T1204": "TA0002", "T1204.001": "TA0002", "T1204.002": "TA0002",
    "T1203": "TA0002",
    "T1059": "TA0002", "T1059.001": "TA0002", "T1059.003": "TA0002",
    "T1059.005": "TA0002", "T1059.006": "TA0002", "T1059.007": "TA0002",
    "T1106": "TA0002",
    "T1053": "TA0003", "T1053.005": "TA0003",
    "T1547": "TA0003", "T1547.001": "TA0003",
    "T1543": "TA0003", "T1543.003": "TA0003",
    "T1078": "TA0004", "T1078.002": "TA0004",
    "T1055": "TA0004", "T1055.001": "TA0004", "T1055.012": "TA0004",
    "T1548": "TA0004", "T1548.002": "TA0004",
    "T1574": "TA0004", "T1574.001": "TA0004", "T1574.002": "TA0004",
    "T1112": "TA0005", "T1140": "TA0005", "T1027": "TA0005",
    "T1027.002": "TA0005", "T1027.010": "TA0005",
    "T1070": "TA0005", "T1070.004": "TA0005",
    "T1036": "TA0005", "T1036.005": "TA0005",
    "T1127": "TA0005", "T1127.001": "TA0005",
    "T1218": "TA0005", "T1218.011": "TA0005",
    "T1562": "TA0005", "T1562.001": "TA0005",
    "T1553": "TA0005", "T1553.002": "TA0005",
    "T1110": "TA0006", "T1110.003": "TA0006",
    "T1003": "TA0006", "T1003.001": "TA0006",
    "T1555": "TA0006", "T1555.003": "TA0006",
    "T1552": "TA0006", "T1552.001": "TA0006",
    "T1082": "TA0007", "T1083": "TA0007", "T1018": "TA0007",
    "T1057": "TA0007", "T1012": "TA0007", "T1016": "TA0007",
    "T1049": "TA0007", "T1007": "TA0007", "T1033": "TA0007",
    "T1069": "TA0007", "T1087": "TA0007", "T1135": "TA0007",
    "T1021": "TA0008", "T1021.001": "TA0008", "T1021.002": "TA0008",
    "T1570": "TA0008", "T1080": "TA0008",
    "T1105": "TA0011", "T1071": "TA0011", "T1071.001": "TA0011",
    "T1071.004": "TA0011", "T1219": "TA0011",
    "T1573": "TA0011", "T1573.001": "TA0011", "T1573.002": "TA0011",
    "T1008": "TA0011", "T1095": "TA0011", "T1102": "TA0011",
    "T1132": "TA0011", "T1132.001": "TA0011",
    "T1074": "TA0009", "T1074.001": "TA0009",
    "T1560": "TA0009", "T1560.001": "TA0009",
    "T1119": "TA0009", "T1115": "TA0009",
    "T1005": "TA0009", "T1213": "TA0009",
    "T1041": "TA0010", "T1048": "TA0010", "T1567": "TA0010",
    "T1486": "TA0040", "T1490": "TA0040", "T1489": "TA0040",
    "T1485": "TA0040", "T1529": "TA0040",
}

def get_tactic_id(technique_id: str) -> str | None:
    if technique_id in TECHNIQUE_TACTIC_MAP:
        return TECHNIQUE_TACTIC_MAP[technique_id]
    parent = technique_id.split(".")[0]
    return TECHNIQUE_TACTIC_MAP.get(parent)

def get_tactic_name(tactic_id: str) -> str:
    return TACTIC_NAMES.get(tactic_id, "Unknown")

def get_phase_index(tactic_id: str | None) -> int:
    if tactic_id is None:
        return 99
    return TACTIC_ORDER.get(tactic_id, 99)

def technique_phase(technique_id: str) -> int:
    tactic = get_tactic_id(technique_id)
    return get_phase_index(tactic)
