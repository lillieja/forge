#include <Servo.h>

Servo myServo;

void setup() {
  Serial.begin(9600); 
  myServo.attach(9);  
  
  // Set to initial rest position
  myServo.write(90); 
}

void loop() {
  if (Serial.available() > 0) {
    // Read the integer
    int angle = Serial.parseInt();
    
    // Check for a valid angle
    if (angle >= 0 && angle <= 180) {
      myServo.write(angle);
      
      // CRITICAL: Clear the remaining buffer (like '\n' or '\r')
      // This prevents the code from reading a "0" on the next pass
      while (Serial.available() > 0) {
        Serial.read();
      }
    }
  }
}
