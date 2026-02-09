from cv2_enumerate_cameras import enumerate_cameras

def list_cameras():
    cams = enumerate_cameras()
    print("=== Camera List ===")

    for cam in cams:
        print(
            f"[index={cam.index}] "
            f"name='{cam.name}' "
            f"vid={cam.vid} pid={cam.pid}"        
        )

if __name__ == "__main__":
    list_cameras()
