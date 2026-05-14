from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import threading

app = FastAPI()

bins = [
    {"id": "A1", "current": 0, "total": 5, "status": "white"},
    {"id": "A2", "current": 0, "total": 3, "status": "white"},
    {"id": "A3", "current": 0, "total": 20, "status": "white"},
    {"id": "A4", "current": 0, "total": 0, "status": "grey"},
    {"id": "B1", "current": 0, "total": 0, "status": "grey"},
    {"id": "B2", "current": 0, "total": 10, "status": "white"},
    {"id": "B3", "current": 0, "total": 0, "status": "grey"},
]

valid_statuses = {"white", "green", "orange", "red", "grey"}


@app.get("/api/bins")
def get_bins():
    return bins


@app.get("/")
def index():
    return FileResponse("static/index.html")


def find_bin(bin_id):
    for bin in bins:
        if bin["id"].lower() == bin_id.lower():
            return bin
    return None


def terminal_loop():
    print("\nBin monitor terminal controls")
    print("Commands:")
    print("  set A1 5")
    print("  clear A1")
    print("  show")
    print("  help")
    print()

    while True:
        command = input("> ").strip()
        parts = command.split()

        if not parts:
            continue

        action = parts[0].lower()

        if action == "help":
            print("Commands:")
            print("  set <bin_id> <current> <total>")
            print("  clear <bin_id>")
            print("  show")
            continue

        if action == "show":
            for bin in bins:
                print(bin)
            continue

        if action == "clear":
            if len(parts) != 2:
                print("Use: clear A1")
                continue

            bin_id = parts[1]
            bin = find_bin(bin_id)

            if bin is None:
                print(f"Unknown bin: {bin_id}")
                continue

            bin["current"] = 0
            bin["total"] = 0
            bin["status"] = "grey"
            print(f"{bin_id} cleared")
            continue

        if action == "set":
            if len(parts) != 3:
                print("Use: set A1 5")
                continue

            bin_id = parts[1]
            current_text = parts[2]

            bin = find_bin(bin_id)

            if bin is None:
                print(f"Unknown bin: {bin_id}")
                continue

            try:
                current = int(current_text)
                total = bin["total"]
                if current == 0 and total != 0:
                    status = "white"
                elif current < total and current != 0:
                    status = "orange"
                elif current == total:
                    status = "green"
                elif current > total:
                    status = "red"
                else:
                    status = "grey"

            except ValueError:
                print("Current and total must be numbers")
                continue

            if current < 0 or total < 0:
                print("Current and total cannot be negative")
                continue

            bin["current"] = current
            bin["status"] = status

            print(f"{bin_id} updated: {current}/{total}")
            continue

        print("Unknown command. Type help.")
        

app.mount("/static", StaticFiles(directory="static"), name="static")


threading.Thread(target=terminal_loop, daemon=True).start()