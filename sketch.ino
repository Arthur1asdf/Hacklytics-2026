void setup() {
  // setup port
  Serial.begin(9600);

  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(8, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);
  pinMode(11, OUTPUT);

  
  digitalWrite(LED_BUILTIN, LOW);
  digitalWrite(8, LOW);
  digitalWrite(9, LOW);
  digitalWrite(10, LOW);
  digitalWrite(11, LOW);
}

void loop() {
  // if (Serial.available()) {
  //   char c = Serial.read();
  //   if (c == 'left_arm') {
  //     digitalWrite(8, HIGH);
  //     delay(1000);
  //     digitalWrite(8, LOW);
  //     delay(1000);
  //   }
  //   if (c == 'right_arm'){
  //     digitalWrite(9, HIGH);
  //     delay(1000);
  //     digitalWrite(9, LOW);
  //     delay(1000);
  //   }
  //   if (c == 'left_leg'){
  //     digitalWrite(10, HIGH);
  //     delay(1000);
  //     digitalWrite(10, LOW);
  //     delay(1000);
  //   }
  //   if(c == 'right_leg'){
  //     digitalWrite(11, HIGH);
  //     delay(1000);
  //     digitalWrite(11, LOW);
  //     delay(1000);
  //   }
  //   if (c == 'chest'){
  //     digitalWrite(8, HIGH);
  //     digitalWrite(9, HIGH);
  //     delay(1000);
  //     digitalWrite(8, LOW);
  //     digitalWrite(9, LOW);
  //     delay(1000);
  //   }
  //   if(c == 'torso'){
  //     digitalWrite(10, HIGH);
  //     digitalWrite(11, HIGH);
  //     delay(1000);
  //     digitalWrite(10, LOW);
  //     digitalWrite(11, LOW);
  //     delay(1000);
  //   }
  // }
  String c = "left_arm";
    if (c == "left_arm") {
      digitalWrite(8, HIGH);
      delay(2000);
      digitalWrite(8, LOW);
      delay(1300);
    }
    if (c == "right_arm"){
      digitalWrite(9, HIGH);
      delay(2000);
      digitalWrite(9, LOW);
      delay(1300);
    }
    if (c == "left_leg"){
      digitalWrite(10, HIGH);
      delay(2000);
      digitalWrite(10, LOW);
      delay(1300);
    }
    if(c == "right_leg"){
      digitalWrite(11, HIGH);
      delay(2000);
      digitalWrite(11, LOW);
      delay(1300);
    }
    if (c == "chest"){
      digitalWrite(8, HIGH);
      digitalWrite(9, HIGH);
      delay(1000);
      digitalWrite(8, LOW);
      digitalWrite(9, LOW);
      delay(1000);
    }
    if(c == "torso"){
      digitalWrite(10, HIGH);
      digitalWrite(11, HIGH);
      delay(1000);
      digitalWrite(10, LOW);
      digitalWrite(11, LOW);
      delay(1000);
    }
}