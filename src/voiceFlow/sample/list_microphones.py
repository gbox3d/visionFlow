import sounddevice as sd

def list_microphones():
    """
    시스템에 연결된 마이크 (입력 장치) 목록을 출력합니다.
    """
    print("사용 가능한 오디오 입력 장치:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']} (API: {sd.query_hostapis(device['hostapi'])['name']})")
    print("기본 입력 장치:")
    try:
        default_input_device_info = sd.query_devices(kind='input')
        print(f"  {default_input_device_info['index']}: {default_input_device_info['name']} (API: {sd.query_hostapis(default_input_device_info['hostapi'])['name']})")
    except Exception as e:
        print(f"  기본 입력 장치를 찾을 수 없습니다: {e}")

if __name__ == "__main__":
    list_microphones()
