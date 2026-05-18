from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json

app = FastAPI()
@app.get("/api/bins")
def get_bins():
    with open("../bins.json", "r") as file:
        bin_data = json.load(file)
    with open("../hand_position.json", "r") as file:
        hand_data = json.load(file)

    bins = bin_data["bins"]
    hand_pos = hand_data["hand_over_bin"]

    for bin in bins:
        id = bin["id"]
        current = int(bin["current"])
        total = int(bin["total"])
        using = bin["using"].lower() == "true"  

        if using:
            bin["status"] = calculate_status(current, total)
        else:
            if hand_pos == id:
                bin["status"] = "wrong_bin"
            else:
                bin["status"] = "grey"

    return bins

@app.get("/api/mainBin")
def get_mainBin():
    with open("../mainBin.json", "r") as file:
        mainBin_data = json.load(file)

    return mainBin_data

@app.get("/")
def index():
    return FileResponse("static/index.html")


def find_bin(bin_id):
    for bin in bins:
        if bin["id"].lower() == bin_id.lower():
            return bin
    return none

def calculate_status(current, total):
    if current == 0:
        return "white"
    elif current < total:
        return "orange"
    elif current == total:
        return "green"
    else:
        return "warn"

app.mount("/static", StaticFiles(directory="static"), name="static")