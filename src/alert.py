def build_alert_message(alert_count: int) -> str:
    if alert_count <= 0:
        return "안전모 미착용 위험이 감지되지 않았습니다."
    return f"안전모 미착용 작업자 {alert_count}명이 감지되었습니다. 안전모를 착용하세요."


def reason_for_detection(class_name: str) -> str:
    if class_name == "no_helmet":
        return "머리 영역이 보이지만 안전모가 감지되지 않았습니다."
    if class_name == "no_helmet_person":
        return "사람 상단 영역에 안전모가 감지되지 않았습니다. 일반 캡/모자는 안전모로 인정하지 않습니다."
    if class_name == "helmet_rejected":
        return "안전모로 탐지됐지만 크기/형태가 실제 안전모 후보로 보기 어려워 보호구로 인정하지 않았습니다."
    if class_name == "cap_hat":
        return "일반 캡/모자는 안전모로 인정하지 않습니다. 안전모 착용이 필요합니다."
    if class_name == "helmet":
        return "안전모로 판단되는 보호구가 감지되었습니다."
    return "추가 확인이 필요한 탐지 결과입니다."
