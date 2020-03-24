print("welcome heros world !")
print('input you name:')
name = input()
hp = 100
if not name:
    name = 'player01'

usermsg = [name, hp]
print("your hero's name is :", usermsg[0], "Hp is :", usermsg[1])
print("you are here: ##*## |'a' for left ,'d' for right |")
userinput = input()
if userinput == 'a':
    print("you are here: *#### |'a' for left ,'d' for right |")
if userinput == 'd':
    print("you are here: ####* |'a' for left ,'d' for right |")
