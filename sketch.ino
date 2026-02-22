void setup() {
  Serial.begin(9600);
  pinMode(8, HIGH);
  pinMode(9, HIGH);
  pinMode(10, HIGH);
  pinMode(11, HIGH);
}

void runStatus(char stat, int pin) {
  if (stat == '0') { // GREEN
    digitalWrite(pin, HIGH);
    //delay(1000);
  } else if (stat == '1') {
    digitalWrite(pin, HIGH);
    delay(500);
    digitalWrite(pin, LOW);
    delay(500);
  } else if (stat == '2') {
    digitalWrite(pin, HIGH);
    delay(500);
    digitalWrite(pin, LOW);
    delay(500);
  } else if (stat == '3') {
    digitalWrite(pin, HIGH);
    delay(500);
    digitalWrite(pin, LOW);
    delay(500);
  }
}

void loop() {
  if (Serial.available() >= 2) {
    char limbChar = Serial.read();
    char statusChar = Serial.read(); 

    // Blink the LED to show we received something
    switch (limbChar) {
      case '0': // left arm
        runStatus(statusChar, 8);
        break;
      case '1': // right arm
        runStatus(statusChar, 9);
        break;
      case '2': // left leg
        runStatus(statusChar, 10);
        break;
      case '3': // right leg
        runStatus(statusChar, 11);
        break;
    }
  }
  digitalWrite(8, LOW);
}