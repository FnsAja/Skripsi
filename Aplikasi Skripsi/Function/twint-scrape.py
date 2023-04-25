import twint

t = twint.Config()

t.Search = "Kanker"
t.Store_csv = True
t.Output = "Kanker.csv"
t.Limit = 10

try:
    twint.run.Search(t)
except Exception as e:
    print(e)

print("end of the program")