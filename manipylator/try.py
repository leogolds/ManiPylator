from tasks import detect_hands_latest

task = detect_hands_latest("tcp://127.0.0.1:5555")
print(task.get(blocking=True))
