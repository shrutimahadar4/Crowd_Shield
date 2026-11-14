import sys
sys.path.append("D:\\CrowdShield - v22.02.26\\crowd-dashboard")
from cap_alert import dispatch_cap_alert

print("Running Mock CAP Generator...")
success = dispatch_cap_alert(
    alert_id="MOCK-ALERT-999-XYZ",
    count=145,
    capacity=50,
    location="Sector 7G (Main Entrance)",
    label="HIGH EMERGENCY"
)

if success:
    print("Test passed! Valid XML payload delivered to console.")
else:
    print("Test failed.")
