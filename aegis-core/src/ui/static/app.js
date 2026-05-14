async function loadBins() {
  try {
    const response = await fetch("/api/bins");

    if (!response.ok) {
      throw new Error("Backend error");
    }

    const bins = await response.json();
    renderBins(bins);

    document.getElementById("last-updated").textContent =
      "Last updated: " + new Date().toLocaleTimeString();
  } catch (error) {
    console.error(error);

    document.getElementById("last-updated").textContent =
      "Could not reach backend";
  }
}

function renderBins(bins) {
  const container = document.getElementById("bins");
  container.innerHTML = "";

  for (const bin of bins) {
    const box = document.createElement("div");

    const status = String(bin.status).trim().toLowerCase();

    box.className = `bin ${status}`;

    if (status !== "grey") {
      box.innerHTML = `
        <div class="bin-id">${bin.id}</div>
        <div class="quantity">${bin.current}/${bin.total}</div>
      `;
    }

    // Important: this happens for ALL bins, including grey ones
    container.appendChild(box);
  }
}

loadBins();
setInterval(loadBins, 2000);
