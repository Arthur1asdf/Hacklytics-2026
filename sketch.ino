// One pin per limb: green = off, yellow = blink, red = on. Other modes = off.
// Limb indices: 0 = left arm, 1 = right arm, 2 = left leg, 3 = right leg
// State: '0' = green (off), '1' = yellow (blink), '2' = red (on), '3'/'4'/'5' = off

const int PINS[] = {8, 9, 10, 11};  // left arm, right arm, left leg, right leg
const int NUM_LIMBS = 4;

// Stored state per limb: 0=green(off), 1=yellow(blink), 2=red(on), 3+=off
char limbState[NUM_LIMBS] = {'0', '0', '0', '0'};

const unsigned long BLINK_MS = 500;

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < NUM_LIMBS; i++) {
    pinMode(PINS[i], OUTPUT);
    digitalWrite(PINS[i], LOW);
  }
}

void loop() {
  // Read any new limb/status from Python (2 bytes: limb index, state)
  if (Serial.available() >= 2) {
    char limbChar = Serial.read();
    char statusChar = Serial.read();
    int idx = limbChar - '0';
    if (idx >= 0 && idx < NUM_LIMBS) {
      limbState[idx] = statusChar;
    }
  }

  unsigned long t = millis();
  int blinkOn = (t / BLINK_MS) % 2;

  for (int i = 0; i < NUM_LIMBS; i++) {
    int pin = PINS[i];
    char s = limbState[i];

    if (s == '0') {
      // Green: off
      digitalWrite(pin, LOW);
    } else if (s == '1') {
      // Yellow: blink
      digitalWrite(pin, blinkOn ? HIGH : LOW);
    } else if (s == '2') {
      // Red: constantly on
      digitalWrite(pin, HIGH);
    } else {
      // Missing / Occluded / Unknown: off
      digitalWrite(pin, LOW);
    }
  }
}
