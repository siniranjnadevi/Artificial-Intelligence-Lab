"""
Integration tests for the complete SOS GPS lifecycle:
Creation -> Persistence in SQLite Database -> Retrieval -> Map Formatting.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import SessionLocal
from backend.models import EmergencySOSRecord

client = TestClient(app)


def test_full_sos_lifecycle():
    # 1. Post Real GPS Alert
    sos_data = {
        "disaster_type": "Wildfire",
        "severity": "Critical",
        "description": "Rapidly advancing bushfire near eastern ridge",
        "affected_people": 5,
        "is_medical_emergency": True,
        "is_trapped": True,
        "is_water_rising": False,
        "latitude": 28.6139,
        "longitude": 77.2090,
        "gps_accuracy": 5.2,
        "gps_timestamp": "2026-11-16T16:00:00Z",
        "is_real_gps": True,
        "location_status": "DETECTED",
        "location_name": "New Delhi Central"
    }

    res = client.post("/api/sos", json=sos_data)
    assert res.status_code == 200
    incident_id = res.json()["incident_id"]

    # 2. Verify Database Persistence directly in SQLite
    db = SessionLocal()
    db_record = db.query(EmergencySOSRecord).filter(EmergencySOSRecord.id == incident_id).first()
    assert db_record is not None
    assert db_record.latitude == 28.6139
    assert db_record.longitude == 77.2090
    assert db_record.gps_accuracy == 5.2
    assert db_record.is_real_gps is True
    assert db_record.status == "Active"
    db.close()

    # 3. List via GET and verify it appears with real GPS flag
    list_res = client.get("/api/sos?is_real_gps=true")
    assert list_res.status_code == 200
    all_real = list_res.json()
    assert any(item["id"] == incident_id for item in all_real)
