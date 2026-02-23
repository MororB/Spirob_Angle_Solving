#include <Adafruit_MLX90393.h>
#include <Arduino.h>
#include <Wire.h>


// ================== PIN KONFIGURATION ==================
#define SDA_PIN 21
#define SCL_PIN 22
#define I2C_SPEED 400000

// ================== SENSOR GRUNDADRESSEN ==================
// Diese Adressen werden vom LTC4316 mit XOR modifiziert
const uint8_t BASE_MMA = 0x1C; // MMA8452Q Grundadresse
const uint8_t BASE_MLX = 0x0C; // MLX90393 Grundadresse

// ================== MMA8452Q REGISTER ==================
#define MMA_REG_WHO_AM_I 0x0D
#define MMA_REG_OUT_X_MSB 0x01
#define MMA_REG_XYZ_DATA_CFG 0x0E
#define MMA_REG_CTRL_REG1 0x2A
#define MMA_WHO_AM_I_VALUE 0x2A // Erwarteter Wert

// ================== ERKANNTE PLATINEN ==================
#define MAX_BOARDS 16
uint8_t foundXorIds[MAX_BOARDS];
int numFoundBoards = 0;

// ================== TIMING ==================
#define FRAME_INTERVAL_US 20000  // 50 Hz: 1,000,000 µs / 50 = 20,000 µs

// ================== MLX OBJEKTE ==================
// Wir brauchen ein Array von MLX-Objekten fuer jede Platine
Adafruit_MLX90393 mlxSensors[MAX_BOARDS];
bool mlxInitialized[MAX_BOARDS] = {false};

// ================== FUNKTIONEN ==================

/**
 * Prueft ob ein I2C-Geraet an der Adresse antwortet
 */
bool devicePresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return (Wire.endTransmission() == 0);
}

/**
 * Prueft ob an der Adresse ein MMA8452Q ist (WHO_AM_I Check)
 */
bool isMMA8452Q(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(MMA_REG_WHO_AM_I);
  if (Wire.endTransmission(false) != 0)
    return false;

  if (Wire.requestFrom(addr, (uint8_t)1) != 1)
    return false;

  uint8_t whoAmI = Wire.read();
  return (whoAmI == MMA_WHO_AM_I_VALUE);
}

/**
 * Konfiguriert einen MMA8452Q Sensor
 */
bool setupMMA(uint8_t addr) {
  if (!devicePresent(addr))
    return false;

  // Standby Mode
  Wire.beginTransmission(addr);
  Wire.write(MMA_REG_CTRL_REG1);
  Wire.write(0x00);
  Wire.endTransmission();

  // +/- 4g Range
  Wire.beginTransmission(addr);
  Wire.write(MMA_REG_XYZ_DATA_CFG);
  Wire.write(0x01);
  Wire.endTransmission();

  // Active Mode, 800Hz Data Rate
  Wire.beginTransmission(addr);
  Wire.write(MMA_REG_CTRL_REG1);
  Wire.write(0x09);
  Wire.endTransmission();

  return true;
}

/**
 * Liest Beschleunigungsdaten vom MMA8452Q
 */
bool readMMA(uint8_t addr, float &ax, float &ay, float &az) {
  Wire.beginTransmission(addr);
  Wire.write(MMA_REG_OUT_X_MSB);
  if (Wire.endTransmission(false) != 0)
    return false;

  if (Wire.requestFrom(addr, (uint8_t)6) != 6)
    return false;

  int16_t x = (Wire.read() << 8) | Wire.read();
  int16_t y = (Wire.read() << 8) | Wire.read();
  int16_t z = (Wire.read() << 8) | Wire.read();

  // 12-bit Daten, +/- 4g Range -> 1g = 512 counts
  ax = (float)(x >> 4) / 512.0;
  ay = (float)(y >> 4) / 512.0;
  az = (float)(z >> 4) / 512.0;

  return true;
}

/**
 * Scannt den I2C-Bus und findet alle Platinen
 * Erkennt die XOR-ID durch Vergleich mit Grundadressen
 */
void scanForBoards() {
  Serial.println("INFO,Scanne nach Platinen...");
  numFoundBoards = 0;

  // Scanne alle moeglichen XOR-Werte (0-127)
  // LTC4316 kann 7-bit Adressen uebersetzen
  for (uint8_t xorId = 0; xorId < 128 && numFoundBoards < MAX_BOARDS; xorId++) {
    uint8_t mmaAddr = BASE_MMA ^ xorId;
    uint8_t mlxAddr = BASE_MLX ^ xorId;

    // Ueberspringe reservierte Adressen
    if (mmaAddr < 0x08 || mmaAddr > 0x77)
      continue;
    if (mlxAddr < 0x08 || mlxAddr > 0x77)
      continue;

    // Pruefe ob MMA vorhanden ist
    if (isMMA8452Q(mmaAddr)) {
      // Gefunden! Speichere XOR-ID
      foundXorIds[numFoundBoards] = xorId;

      // MMA konfigurieren
      setupMMA(mmaAddr);

      // MLX initialisieren
      mlxInitialized[numFoundBoards] =
          mlxSensors[numFoundBoards].begin_I2C(mlxAddr, &Wire);
      if (mlxInitialized[numFoundBoards]) {
        mlxSensors[numFoundBoards].setGain(MLX90393_GAIN_1X);
        mlxSensors[numFoundBoards].setFilter(MLX90393_FILTER_2);
        mlxSensors[numFoundBoards].setOversampling(MLX90393_OSR_0);
      }

      Serial.print("INFO,Platine gefunden: XOR-ID=");
      Serial.print(xorId);
      Serial.print(" MMA=0x");
      Serial.print(mmaAddr, HEX);
      Serial.print(" MLX=0x");
      Serial.print(mlxAddr, HEX);
      Serial.print(" MLX-OK=");
      Serial.println(mlxInitialized[numFoundBoards] ? "ja" : "nein");

      numFoundBoards++;
    }
  }

  Serial.print("INFO,Scan abgeschlossen. Gefundene Platinen: ");
  Serial.println(numFoundBoards);
}

// ================== BINARY FRAME FORMAT ==================
// [0xAA,0x55] + frame_id(uint32) + t_us(uint32) + num_sensors(uint8)
// then for each sensor:
// [sensor_id(uint8)] + [accX,accY,accZ,magX,magY,magZ] as 6 floats (24 bytes)
#define FRAME_HDR_0 0xAA
#define FRAME_HDR_1 0x55

struct __attribute__((packed)) SensorPacket {
  uint8_t sensor_id;
  float accX;
  float accY;
  float accZ;
  float magX;
  float magY;
  float magZ;
};
static_assert(sizeof(SensorPacket) == 1 + 6 * 4, "SensorPacket size mismatch");

// ================== DEBUG ==================
#define DEBUG_SERIAL 0

// ================== SETUP ==================
void setup() {
  Serial.begin(1000000);
  delay(100);

  if (DEBUG_SERIAL) {
    Serial.println("INFO,=== Sensor System Start ===");
    Serial.println("INFO,Initialisiere I2C...");
  }

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(I2C_SPEED);

  delay(100); // Warten bis Sensoren bereit

  // Automatischer Scan
  scanForBoards();

  if (DEBUG_SERIAL) {
    if (numFoundBoards == 0) {
      Serial.println("WARN,Keine Platinen gefunden!");
    }

    Serial.println("INFO,=== Starte Datenausgabe ===");
  }
}

// ================== LOOP ==================
void loop() {
  static uint32_t frame_id = 0;
  static uint32_t last_frame_time = 0;
  
  // Warte bis zum nächsten Frame Intervall
  uint32_t now = micros();
  if (now - last_frame_time < FRAME_INTERVAL_US) {
    delayMicroseconds(FRAME_INTERVAL_US - (now - last_frame_time));
    now = micros();
  }
  last_frame_time = now;
  
  const uint32_t t_us = now;

  // Header senden
  uint8_t header[2] = {FRAME_HDR_0, FRAME_HDR_1};
  Serial.write(header, 2);
  Serial.write((uint8_t *)&frame_id, sizeof(frame_id));
  Serial.write((uint8_t *)&t_us, sizeof(t_us));

  // Anzahl Sensoren senden
  uint8_t n = (uint8_t)numFoundBoards;
  Serial.write(&n, 1);

  // Alle gefundenen Platinen auslesen und als Binary senden
  for (int i = 0; i < numFoundBoards; i++) {
    uint8_t xorId = foundXorIds[i];
    uint8_t mmaAddr = BASE_MMA ^ xorId;

    // Beschleunigung lesen
    float ax = 0, ay = 0, az = 0;
    readMMA(mmaAddr, ax, ay, az);

    // Magnetfeld lesen
    float mx = 0, my = 0, mz = 0;
    if (mlxInitialized[i]) {
      mlxSensors[i].readData(&mx, &my, &mz);
    }

    SensorPacket pkt;
    pkt.sensor_id = xorId;
    pkt.accX = ax;
    pkt.accY = ay;
    pkt.accZ = az;
    pkt.magX = mx;
    pkt.magY = my;
    pkt.magZ = mz;

    Serial.write((uint8_t *)&pkt, sizeof(pkt));
  }

  frame_id++;
}