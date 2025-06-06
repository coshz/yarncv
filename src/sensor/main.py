import serial
import serial.rs485

ser = serial.Serial()
ser.rs485_mode = serial.rs485.RS485Settings(
)