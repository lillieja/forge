#include <Servo.h>

Servo myServo;

void setup() {
  Serial.begin(9600); // Open serial connection at 9600 baud
  myServo.attach(9);  // Assumes servo signal is on Pin 9
}

void loop() {
  if (Serial.available() > 0) {
    // Read the incoming byte (0-180)
    int angle = Serial.parseInt(); 
    
    // Safety check to ensure the angle is valid
    if (angle >= 0 && angle <= 180) {
      myServo.write(angle);
    }
  }
}