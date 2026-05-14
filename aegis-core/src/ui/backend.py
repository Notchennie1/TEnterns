from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import threading
import json

app = FastAPI()
@app.get("/api/bins")
def get_bins():
    with open("../bins.json", "r") as file:
        data = json.load(file)

    bins = data["bins"]

    for bin in bins:
        current = int(bin["current"])
        total = int(bin["total"])
        if bin["status"] != "grey":
            bin["status"] = calculate_status(current, total)

    return bins


@app.get("/")
def index():
    return FileResponse("static/index.html")


def find_bin(bin_id):
    for bin in bins:
        if bin["id"].lower() == bin_id.lower():
            return bin
    return None

def calculate_status(current, total):
    if total == 0:
        return "grey"
    elif current == 0:
        return "white"
    elif current < total:
        return "orange"
    elif current == total:
        return "green"
    else:
        return "red"

app.mount("/static", StaticFiles(directory="static"), name="static")