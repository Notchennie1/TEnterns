/* ============================================================
   3-Cell Load Cell Scale — Medium Bin
   - Reads 3x HX711 amplifiers (one per load cell)
   - Prints each cell reading (grams)
   - Prints total weight  = F1 + F2 + F3   (position-independent)
   - Prints load centroid = weighted position (the "triangulation")
   ============================================================ */

#include "HX711.h"

// ---------- PIN ASSIGNMENTS (change to your wiring) ----------
// Each HX711 needs a DOUT (data) and SCK (clock) pin.
const int DOUT_PIN[3] = {18, 19, 21};
const int SCK_PIN  = 5;

HX711 cell[3];

// ---------- CALIBRATION (REPLACE WITH YOUR VALUES) ----------
// scale  = raw counts per gram  (from calibration)
// offset = raw count at zero load (tare)
// Run calibrate() once, read the values from Serial, then paste here
// and set DO_CALIBRATION = false.
float cellScale[3]  = { 1054.8583, 1076.3196, 1007.3719 };   // <-- counts per gram, per cell
long  cellOffset[3] = { 50341, -126173, -58017 };   // <-- tare offset, per cell

const bool DO_CALIBRATION = false;   // set false once you have real values
const float CAL_MASS_G   = 183.4;   // known reference mass you'll use, grams

// ---------- CELL POSITIONS (mm, from platform centre) ----------
// Default = ideal R69 triangle, 120deg apart, cell 1 along +Y.
// REPLACE with your measured (x,y) if your layout is different.
const float cellX[3] = {   0.0f, 29.87f,  -29.87f };
const float cellY[3] = {  34.5f, -17.25f,  -17.25f  };

// ---------- FILTER ----------
const int N_SAMPLES = 10;   // readings averaged per cell per cycle

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  for (int i = 0; i < 3; i++) {
    cell[i].begin(DOUT_PIN[i], SCK_PIN);
  }

  Serial.println(F("3-cell scale starting..."));

  if (DO_CALIBRATION) {
    calibrate();
  } else {
    // Apply stored counts-per-gram for each cell.
    for (int i = 0; i < 3; i++) {
      cell[i].set_scale(cellScale[i]);
    }
    // Re-tare at boot instead of trusting frozen offsets. Load-cell zero
    // drifts with temperature, mounting stress, creep and supply voltage,
    // so a hardcoded offset reads non-zero at no load. Make sure the
    // platform is EMPTY at power-up.
    Serial.println(F("Remove ALL load. Taring in 5s..."));
    delay(5000);
    for (int i = 0; i < 3; i++) {
      cell[i].tare(20);                 // average 20 readings as the new zero
      cellOffset[i] = cell[i].get_offset();
    }
    Serial.println(F("Boot tare done. Using stored scale."));
  }

  Serial.println(F("Total_g, cell1_g,cell2_g,cell3_g"));
}

void loop() {
  float g[3];
  float total = 0;

  // --- read each cell in grams ---
  for (int i = 0; i < 3; i++) {
    // get_units() returns (raw - offset) / scale  = grams
    g[i] = cell[i].get_units(N_SAMPLES);
    total += g[i];
  }

  // --- total weight = simple sum (independent of position) ---
  // This is the number to use for your delta-mass kitting logic.

  // --- load centroid ("triangulation") ---
  // Weighted average of cell positions by the force each carries.
  // Only valid when total is meaningfully > 0.
  float cx = 0, cy = 0;
  if (total > 1.0) {                 // ignore noise near zero
    for (int i = 0; i < 3; i++) {
      cx += g[i] * cellX[i];
      cy += g[i] * cellY[i];
    }
    cx /= total;
    cy /= total;
  }

  // --- output as CSV ---
  Serial.println(total, 2); Serial.print(',');
  Serial.print(g[0], 2); Serial.print(',');
  Serial.print(g[1], 2); Serial.print(',');
  Serial.println(g[2], 2); Serial.print(',');

  delay(200);   // ~5 Hz output
}

/* ============================================================
   CALIBRATION ROUTINE  (per-cell, position-independent)

   Rationale: the total weight F1+F2+F3 is only position-independent
   if every cell reports the TRUE grams it carries -- i.e. each cell's
   scale must be its own true counts-per-gram. Placing one mass at the
   centre and assuming each cell took 1/3 is wrong (the split is never
   exactly even), which corrupts the per-cell scales and makes the
   total drift with object position. So we calibrate each cell against
   a known mass applied DIRECTLY to that cell, one at a time.

   1. Remove all load when prompted (tare, captures per-cell offset)
   2. For each cell in turn, place CAL_MASS_G directly over THAT cell
      (or hang it from that cell) when prompted
   3. Copy the printed scale/offset values into the arrays above
   4. Set DO_CALIBRATION = false and re-upload
   ============================================================ */
void calibrate() {
  Serial.println(F("\n--- CALIBRATION ---"));
  Serial.println(F("Remove ALL load from platform. Taring in 5s..."));
  delay(5000);

  for (int i = 0; i < 3; i++) {
    cell[i].set_scale();          // scale = 1 for raw
    cell[i].tare(20);             // average 20 readings as zero
    cellOffset[i] = cell[i].get_offset();
  }
  Serial.println(F("Tare done."));

  // --- per-cell span: load each cell individually ---
  for (int i = 0; i < 3; i++) {
    Serial.print(F("\nPlace ")); Serial.print(CAL_MASS_G);
    Serial.print(F(" g DIRECTLY over cell ")); Serial.print(i + 1);
    Serial.println(F(" (only that cell). Measuring in 8s..."));
    delay(8000);

    // raw counts above this cell's own zero, averaged
    long raw = cell[i].read_average(20);
    float countsAboveZero = (float)(raw - cellOffset[i]);
    // true counts-per-gram for this cell -- no 1/3 assumption
    cellScale[i] = countsAboveZero / CAL_MASS_G;
    cell[i].set_scale(cellScale[i]);
    cell[i].set_offset(cellOffset[i]);

    Serial.print(F("  cell ")); Serial.print(i + 1);
    Serial.print(F(" scale = ")); Serial.println(cellScale[i], 4);
  }

  Serial.println(F("\n--- PASTE THESE INTO THE CODE ---"));
  Serial.print(F("float cellScale[3] = { "));
  for (int i = 0; i < 3; i++) {
    Serial.print(cellScale[i], 4);
    Serial.print(i < 2 ? F(", ") : F(" };\n"));
  }
  Serial.print(F("long  cellOffset[3] = { "));
  for (int i = 0; i < 3; i++) {
    Serial.print(cellOffset[i]);
    Serial.print(i < 2 ? F(", ") : F(" };\n"));
  }
  Serial.println(F("Set DO_CALIBRATION = false, then re-upload.\n"));
  delay(3000);
}
