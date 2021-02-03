import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="rootpasswordgiven",
    database="models"
)

mycursor = db.cursor()

mycursor.execute("DESCRIBE runs")

for x in mycursor:
    print(x)
