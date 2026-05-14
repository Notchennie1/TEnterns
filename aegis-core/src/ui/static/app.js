async function loadBins() {
  try {
    const response = await fetch("/api/bins");

    if (!response.ok) {
      throw new Error("Backend error");
    }

    const bins = await response.json();
    renderBins(bins);
    updateWarningBox(bins)

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

function updateWarningBox(bins) {
  const warningOverlay = document.getElementById("warning-overlay");
  const warningContent = document.getElementById("warning-content");

  const redBins = bins.filter((bin) => {
    const status = String(bin.status).trim().toLowerCase();
    return status === "red";
  });

  if (redBins.length > 0) {
    const redBinText = redBins
      .map((bin) => `${bin.id}: ${bin.current}/${bin.total}`)
      .join("<br>");

    warningContent.innerHTML = `
      WARNING<br>
      Red bin detected<br>
      ${redBinText}
    `;

    warningOverlay.classList.remove("hidden");
  } else {
    warningContent.innerHTML = "";
    warningOverlay.classList.add("hidden");
  }
}

loadBins();
setInterval(loadBins, 2000);
