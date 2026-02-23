#include <Arduino.h>
#include <Wire.h>
#include <MLX90393.h>

// ---------------- Konfiguration ----------------
#define BAUD_RATE       1000000      // 1 MBit/s
#define SDA_PIN         21           // ggf. an dein Board anpassen
#define SCL_PIN         22
#define I2C_CLOCK_HZ    400000

#define TCA_ADDR        0x70         // TCA9548A Standardadresse

#define NUM_CHIPS       5            // 5 MLX90393 pro Board
#define N_BOARDS        3            // Anzahl TCA-Kanäle, die Boards enthalten

// ~100 Hz Gesamt (also ~100 Hz pro Board)
#define FRAME_DELAY_US  10000        // 10 ms

// MLX90393-Adressen laut deinem I2C-Scan
uint8_t mlx_addr[NUM_CHIPS] = {0x0C, 0x10, 0x11, 0x12, 0x13};

// Für jedes Board ein Array aus 5 Sensoren
MLX90393 mlx[N_BOARDS][NUM_CHIPS];
MLX90393::txyz data[N_BOARDS][NUM_CHIPS] = {0,0,0,0};


// --------- TCA: Kanal auswählen ---------
void tca_select(uint8_t channel) {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}


// --------- Sende EIN vollständiges Frame ---------
void sendBoardFrame(uint8_t boardIndex) {
  uint8_t header[3] = {0xAA, 0x55, boardIndex};
  Serial.write(header, 3);

  for (int i = 0; i < NUM_CHIPS; i++) {
    Serial.write((uint8_t*)&data[boardIndex][i], sizeof(data[boardIndex][i]));
  }
}


// ---------------- Setup ----------------
void setup() {
  Serial.begin(BAUD_RATE);
  delay(500);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(I2C_CLOCK_HZ);
  delay(10);

  // Alle Boards initialisieren
  for (uint8_t b = 0; b < N_BOARDS; b++) {
    tca_select(b);

    for (uint8_t i = 0; i < NUM_CHIPS; i++) {
      uint8_t status = mlx[b][i].begin(mlx_addr[i], -1, Wire);
      (void)status;
      mlx[b][i].startBurst(0xF); // TXYZ burst mode
    }
  }
}


// ---------------- Loop ----------------
void loop() {
  // Für jedes Board nacheinander lesen & senden
  for (uint8_t b = 0; b < N_BOARDS; b++) {

    // Kanal auswählen
    tca_select(b);

    // Sensoren lesen
    for (int i = 0; i < NUM_CHIPS; i++) {
      mlx[b][i].readBurstData(data[b][i]);
    }

    // Frame senden mit Header + Board ID
    sendBoardFrame(b);
  }

  delayMicroseconds(FRAME_DELAY_US);
}
