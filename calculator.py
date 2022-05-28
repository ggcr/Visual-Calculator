def checkEqual(n):
    if n[-1] == "0":
        return True
    return False

def makeOperation(n):
    return str(eval(n[:-2])) +  " "