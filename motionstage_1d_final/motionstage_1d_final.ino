#include <AccelStepper.h>
#include <MultiStepper.h>

// Function: Print Position (Inline Function)
// Description: Print the Input Coordinate
// Input: x coordinate (int), y coordinate (int)
// Output: N/A
inline void printPosition(int t, int x, int y) {
    char msg[20];
    sprintf(msg, "time = %d, (x, y) = (%d,%d)", t, x/160, y/160);
    Serial.println(msg);
}

// Initialize the two steppers
const int xPulPin = 6;
const int xDirPin = 7;
const int yPulPin = 4;
const int yDirPin = 5;
AccelStepper xStepper(AccelStepper::DRIVER, xPulPin, xDirPin);
AccelStepper yStepper(AccelStepper::DRIVER, yPulPin, yDirPin);

// Create a MultiStepper Group
MultiStepper steppers;

// Record coordinate
long positions[2] = {0,0};

// Record Speed & Time
float speed = 0;
float mmspeed = 0;
float time = 0;
float curx = 0;
float cury = 0;
void setup() {
    // Initialize the Serial Port
    Serial.begin(9600);

    // Add the two steppers to the MultiStpper Group to manage
    steppers.addStepper(xStepper);
    steppers.addStepper(yStepper);

    Serial.println("------------------------------------------");
    Serial.println("Enter 'displacement' [mm]");
    Serial.println("------------------------------------------");
}

void loop() {
  while (Serial.available() > 0) {
    // Serial.println(cury);
    // Parse the user input coordinate
    // time = Serial.parseFloat();
    
    // positions[0] = round(Serial.parseFloat()*160);
    positions[0] = 0;
    positions[1] = round(Serial.parseFloat()*160);

    // Check the maximum distance
    int maxDist = abs(positions[1] - cury);

    // Compute the velocity by the designated time duration
    time = 5;
    speed = maxDist / time;
    // curx = positions[0];
    cury = positions[1];
    // Print the User Input
    printPosition(time, positions[0], positions[1]);

    // Configure each stepper
    xStepper.setMaxSpeed(round(speed));
    yStepper.setMaxSpeed(round(speed));
    // Set the destinations for the steppers
    // positions[0] *= -1; //change to pos if want to reverse xy directions
    steppers.moveTo(positions);
    // Blocks until all are in position
    steppers.runSpeedToPosition();
    while (Serial.available() > 0) Serial.read();
    // Delay to stablize
    delay(1000);
  }
}
