#include "Arduino.h"
#include "util/delay.h"

void delayMs(int ms) {
  // delay is not in ms after PWM change
  unsigned int delayed = ms << 6;
  delay(delayed);
}

void setup() {
  TCCR0B = (TCCR0B & 0b11111000) | 0x01;
  
  Serial.begin(9600);

  pinMode(A0, OUTPUT); // VDD
  pinMode(A1, OUTPUT); // set VSS to output and LOW
  pinMode(A3, OUTPUT); // set EN to output and LOW
  pinMode(5, OUTPUT); // set CTRL to output and LOW, HIGH-Z

  pinMode(2, INPUT);

  Serial.println("Running, perform initial reset:");
  safeMode();  
}

const int delayVal = 1500;
const int ISOdelayVal = 2000;

char rx_byte = 0;
const long rate = 625; // in microseconds
const int duty = 75; // duty cycle for in-chirp modulation

int code1[14] = {1,1,1,1,0,0,0,0,1,1,0,0,1,0};
int code2[10] = {1,0,0,0,1,1,1,0,0,1};
int loopTime = 10000;
int starttime;
int endtime;
int idleCurrentTime;
int idleStartTime;
unsigned long idleTime = 125000 << 6; 
int isSafeMode=1;

void loop() {
  if (Serial.available() > 0) {    // is a character available?
    rx_byte = Serial.read();
    if (rx_byte < '0' || rx_byte > '9') return;
    
    Serial.print("Number received: ");
    Serial.println(rx_byte);

    isSafeMode = 0;
    Serial.println("safe mode 0");
    Serial.print("Switching rate: ");
    Serial.println(rate);
    
    if (rx_byte=='6') { // turn off the whole board
      safeMode();
    } else if (rx_byte=='0') { // turn off the switch
      Serial.println("isolation mode");
      digitalWrite(A0, HIGH);
      delayMs(delayVal);
      digitalWrite(A1, HIGH);
      delayMs(delayVal);
      digitalWrite(A3, HIGH);
    } else if (rx_byte=='1') { //switch to RF1
      Serial.println("switch to RF1 always");
      digitalWrite(A0, HIGH);
      delayMs(delayVal);
      digitalWrite(A1, HIGH);
      delayMs(delayVal);
      digitalWrite(5, HIGH);
      delayMs(delayVal);
      digitalWrite(A3, LOW);
    } else if (rx_byte=='2') { // switch to RF2
      Serial.println("switch to RF2 always");
      digitalWrite(A0, HIGH);
      delayMs(delayVal);
      digitalWrite(A1, HIGH);
      delayMs(delayVal);
      digitalWrite(5, LOW);
      delayMs(delayVal);
      digitalWrite(A3, LOW);
    } else if (rx_byte=='3' || digitalRead(4) ==HIGH) { // switch constantly
      Serial.println("Turning on tag...");
      digitalWrite(A0, HIGH);
      delayMs(delayVal);
      digitalWrite(A1, HIGH);
      delayMs(delayVal);
      digitalWrite(A3, HIGH);
      delayMs(ISOdelayVal);

      Serial.println("Constant switching RF1-RF2");
      starttime = millis();
      endtime = starttime;

      // Switch tag on/off.
      // 1) Use millis(), not delayMicros(), to keep account of
      // the time needed of switching pins. millis() in Arduino
      // Uno is always a multiple of 4us. Overflow of millis()
      // does not matter since the subtraction always provide
      // the correct result.
      // 2) Use direct pin access instead of digitalWrite/Read,
      // since the former takes ~5us and the latter <1us.
      // 3) Tried figuring out a better solution of turning off
      // the tag, but proto-threading (pseudo-multithreading) does
      // not work with Serial input... probably add a button while
      // designing the new 3D tag?
      unsigned long curr;
      digitalWrite(A3, LOW);
      while (digitalRead(2) == LOW) // digitalRead(2) == LOW
      {
        curr = micros();
        PORTD |= _BV(5);
        while (micros() - curr < (rate << 6));
        curr = micros();
        PORTD &= ~_BV(5);
        while (micros() - curr < (rate << 6));
      }
      safeMode();
    }
    else if (rx_byte=='4') { // in-chirp switching
      Serial.println("Turning on tag...");
      digitalWrite(A0, HIGH);
      delayMs(delayVal);
      digitalWrite(A1, HIGH);
      delayMs(delayVal);
      digitalWrite(A3, HIGH);
      delayMs(ISOdelayVal);
      Serial.println("In-chirp modulation at pin 5");

      digitalWrite(A3, LOW);
      analogWrite(5, duty * 255 / 100);

      while (digitalRead(2) == LOW) { // (PIND & _BV(2)) == 0
        if (Serial.available() && Serial.read() == '6') {
          Serial.println("Received 6, stopping in-chirp modulation...");
          break;
        }
      }
      safeMode();
    }
    // Removed old coding modulation 
    else if (rx_byte == '\n') {
      Serial.println("Ignore \\n");
    } else {
      safeMode();
    }
  } else {
    idleCurrentTime = millis();
    if (idleCurrentTime - idleStartTime >= idleTime && isSafeMode == 0) {
      isSafeMode = 1;
      Serial.println("Shutting down after idle...");
      safeMode();
    }
  }
}

void safeMode()
{
  Serial.println("Safe reset in progress...");
  digitalWrite(A3, LOW);
  delayMs(delayVal);
  digitalWrite(5, LOW);
  delayMs(delayVal);
  digitalWrite(A1, LOW);
  delayMs(delayVal);
  digitalWrite(A0, LOW);
  isSafeMode = 1;
  idleStartTime = millis();
  Serial.println("Reset complete");
}
