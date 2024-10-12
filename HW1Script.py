#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Introduction

#P1 : Say "Hello World" With Python 

my_string = "Hello, World!"
print(my_string)

#P2 : Python If-Else

if __name__ == '__main__':
    n = int(input().strip())

    if n%2:
        print('Weird')
    else: 
        if 2 <= n <= 5:
            print("Not Weird")
        if 6 <= n <= 20:
            print("Weird")
        if n > 20:
            print("Not Weird")

#P3: Arithmetic operators 

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    print(a+b)
    print(a-b)
    print(a*b)

#P4: Python : Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    print(a//b)
    print(a/b)


#P5: Loops 

if __name__ == '__main__':
    n = int(input())
    
    for i in range(0,n):
        print(i**2)

#P6: Write a function

def is_leap(year):
    leap = False
    if year%4==0:
        leap=True
        if year%100==0:
            leap=False
            if year%400==0:
                leap=True
    return leap

#P7: Print function

if __name__ == '__main__':
    n = int(input())
    
    for i in range (1,n+1):
        print(i,end='')


# In[ ]:


# Data types

#P1 : Lists
if __name__ == '__main__':
    N = int(input())
    L = []

    for i in range(N):
        instruction = input().split()

        if instruction[0] == 'insert':
            L.insert(int(instruction[1]), int(instruction[2]))
        elif instruction[0] == 'print':
            print(L)
        elif instruction[0] == 'remove':
            if int(instruction[1]) in L:
                L.remove(int(instruction[1]))
        elif instruction[0] == 'append':
            L.append(int(instruction[1]))
        elif instruction[0] == 'sort':
            L.sort()
        elif instruction[0] == 'pop':
            if L:
                L.pop()
        elif instruction[0] == 'reverse':
            L.reverse()
    
#P2: Lists comprehensions 

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    permutations = []
    for i in range(x+1):
        for j in range(y+1):
            for k in range(z+1):
                permutations.append([i,j,k])

    results = []
    for e in permutations:
        if e[0]+e[1]+e[2] != n : 
            results.append(e)

    print(results)

#P3: Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

    arr_list = list(arr)
    L = list(set(arr_list))
    L2 = L.remove(max(L))
    print(max(L))


#P4: Nested Lists

if __name__ == '__main__':
    l=[]
    for _ in range(int(input())):
        name = input()
        score = float(input()) 
        l.append([name,score])
        
        
    min_score = min([k[1] for k in l ])

    l2 = [i for i in l if i[1]!=min_score]

    min_score2= min([k[1] for k in l2 ])

    names = [i[0] for i in l2 if i[1]==min_score2]
    names.sort()
    for i in names :
        print (i)

#P5: Find the percentage 

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for i in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
        
    q_name = input()
    
    result = sum(student_marks[q_name])/3

    result_frmt = "{:.2f}".format(result)

    print(result_frmt)









# In[2]:


# Strings

#P1: Find a string 


#P2: sWAP cASE

def swap_case(s):
    result = ""
    for e in s :
        if e.isupper():
            result+=e.lower()
        elif e.islower():
            result+=e.upper()
        else:
            result+=e
    return result

#P3: String Split and Join

def split_and_join(line):
    a = line.split()
    a = "-".join(a)
    return a

#P4: What's your name

def print_full_name(first, last):
    print(f"Hello {first} {last}! You just delved into python.")


#P5: Mutations

def mutate_string(string, position, character):
    modif = ""
    modif = string[:position] + character + string[position+1:]
    return (modif)


#P6: String Validators

if __name__ == '__main__':
    s = input()
    print(any(char.isalnum() for char in s))
    print(any(char.isalpha() for char in s))
    print(any(char.isdigit() for char in s))
    print(any(char.islower() for char in s))
    print(any(char.isupper() for char in s))


#P7: Text Alignement



#P8: Text Wrap

def wrap(string, max_width):
    div = len(string)// max_width
    reste = len(string)%max_width
    result = ""
    if reste == 0 :
        for k in range(div):
            result += string[max_width*k:max_width*(k+1)] + "\n"
    if reste !=0:
        for k in range(div):
            result +=string[max_width*k:max_width*(k+1)] + "\n"
        result += string[max_width*div:max_width*div + reste] + "\n"
    return(result)

#P9: Designer Door Mat

n, m = map(int, input().split())
for i in range(1, n//2 + 1):
    pattern = ".|." * (2*i - 1)
    print(pattern.center(m, '-'))

print("WELCOME".center(m, '-'))

for i in range(n//2, 0, -1):
    pattern = ".|." * (2*i - 1)
    print(pattern.center(m, '-'))



#P10: String Formatting

def print_formatted(number):
    width = len(bin(number)) - 2
    for i in range(1, number + 1):
        decimal = str(i)
        octal = oct(i)[2:]
        hexadecimal = hex(i)[2:].upper()
        binary = bin(i)[2:]
        
        print(f"{decimal:>{width}} {octal:>{width}} {hexadecimal:>{width}} {binary:>{width}}")

#P11: Alphabet Rangoli


#P12: Capitalize! (!!)

def solve(s):
    words = s.split()
    results=[]
    for word in words:
        if word.isalpha():
            c_word = word[0].upper() + word[1:].lower()
            results.append(c_word)
        else:
            results.append(word)    
    return(' '.join(results))


#P13: The Minion Game


#P14: Merge the Tools!

def merge_the_tools(string, k):
    liste = []
    n = len(string)//k
    for i in range(n):
        s = string[i*k:(i+1)*k]
        result = ""
        seen = set()
        for j in s :
            if j not in seen:
                result+= j
                seen.add(j)
        print(result)
  


# In[ ]:


#Sets 

#P1 : Introduction to Sets

def average(array):
    z = set(array)
    s = 0
    for k in z:
        s+=k
    result = round(s/len(z),3)
    return(result)
 
#P2 : No Idea !

lengths = input()
array = list(input().split())
a = set(input().split())
b = set(input().split())
happiness =0
for k in array:
    if k in a : 
        happiness+=1
    elif k in b : 
        happiness-=1
print(happiness)
        

#P3 : Symmetric Difference

m = input()
M = input()
n = input()
N= input()

a = set(M.split())
b = set(N.split())

setf= a.difference(b).union(b.difference(a))

set_reponse = sorted(map(int,setf))

for i in set_reponse: 
    print(i)


#P4 : Set.add()

n = int(input())
s = set([])
for k in range(n):
    country = input()
    s.add(country)
print(len(s))

#P5 : Set .discard(),.remove(), .pop()

n_set = int(input())
s = set(list(map(int,input().split())))

n= int(input())

for k in range (n):
    ins = list(input().split())
    if ins[0]=="pop":
        if s:
            s.pop()
    elif ins[0]=="remove":
        if int(ins[1]) in s:
            s.remove(int(ins[1]))
    elif ins[0]=="discard":
        s.discard(int(ins[1]))

print(sum(s))

#P6 : Set .union() Operation


n1 = int(input())
s1= set(input().split())
n2 = int(input())
s2 = set(input().split())

s = s1.union(s2)

print(len(s))

#P7 : Set .intersection() Operation

n = input()
a= set(input().split())
m = input()
b = set(input().split())

c = list(a.intersection(b))

print(len(c))


#P8 : Set .difference() Operation

n = input()
a= set(input().split())
m = input()
b = set(input().split())

c = list(a.difference(b))

print(len(c)) 

#P9 : Set .symmetric_difference() Operation

n = input()
a= set(input().split())
m = input()
b = set(input().split())

c = list(a.symmetric_difference(b))

print(len(c)) 

#P10 : Set Mutations

n = int(input())
ensemble_initial = set(map(int, input().split()))
m = int(input())

for _ in range(m):
    donnees_operation = input().split()
    operation = donnees_operation[0]
    longueur_autre_ensemble = int(donnees_operation[1])
    autre_ensemble = set(map(int, input().split()))

    if operation == "intersection_update":
        ensemble_initial.intersection_update(autre_ensemble)
    elif operation == "update":
        ensemble_initial.update(autre_ensemble)
    elif operation == "symmetric_difference_update":
        ensemble_initial.symmetric_difference_update(autre_ensemble)
    elif operation == "difference_update":
        ensemble_initial.difference_update(autre_ensemble)

print(sum(ensemble_initial))

#P11 : The Captain's Room

n = int(input())
s= list(map(int,input().split()))
w= list(set(s))

s1 = sum(w)*n
s2 = sum(s)
print(round((s1 - s2)/(n-1)))

#P12 : Check Subset

n = int(input())
for i in range(n):
    m = int(input())
    A = set(input().split())
    n = int(input())
    B = set(input().split())
    print(A.issubset(B))

#P13 : Check Strict Superset

a = set(map(int, input().split()))
n = int(input())
superset = True

for _ in range(n):
    s= set(map(int, input().split()))   
    if not (a > s):
        superset = False
        break
print(superset)


# In[ ]:


#Collections

#P1 : Collections.Counter()

from collections import Counter
nombre_de_chaussures = int(input())
tailles_de_chaussures = list(map(int, input().split()))
chaussures_disponibles = Counter(tailles_de_chaussures)
nombre_de_clients = int(input())
montant_total = 0
for i in range(nombre_de_clients):
    taille, prix = map(int, input().split())
    if chaussures_disponibles[taille] > 0:
        montant_total += prix
        chaussures_disponibles[taille] -= 1
print(montant_total)

#P2 : Collections.namedtuple()

from collections import namedtuple
import sys

n = int(input())
columns = input().split()
Student = namedtuple('Student',columns)
s=0
for k in range (n):
    x = input().split()
    stud = Student._make(x) 
    s+= int(stud.MARKS)
    
print(s/n)

#P3 : Collections.OrderedDict()

from collections import OrderedDict 

n = int(input())
dico = OrderedDict()
for k in range(n):
    produit = input().strip()
    nom, prix = produit.rsplit(' ', 1)
    if nom in dico :
        dico[nom]+= int(prix)
    else:
        dico[nom]= int(prix)

for nom, total_prix in dico.items():
    print(nom, total_prix)
    
#P4 : Word Order

n = int(input())
words_count={}
words = []

for i in range(n):
    word = input()
    if word not in words_count:
        words.append(word)
        words_count[word] = 1
    else:
        words_count[word] += 1

print(len(words))

for w in words:
   print(words_count[w], end=' ')

#P5 : Collections.deque()


from collections import deque
n = int(input())
d = deque()
for k in range(n):
    instruction = list(input().strip().split())
    if instruction[0] == "append":
        d.append(instruction[1])
    elif instruction[0] == "appendleft":
        d.appendleft(instruction[1])
    elif instruction[0] == "pop":
        if d:
            d.pop()
    elif instruction[0] == "popleft":
        if d:
            d.popleft()

print(" ".join(d))

#P6 : Company Logo

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    s = list(input())
    ls = {}
    
    for l in s:
        if l not in ls:
            ls[l] = 1
        else:
            ls[l] += 1
    
    sorted_ls = sorted(ls.items(), key=lambda i: (-item[1], item[0]))

    for i in range(min(3, len(sorted_ls))):
        cle, valeur = sorted_ls[i]
        print(cle, valeur)


# In[ ]:


# Date and Time : 

#P1: Calendar Module

import calendar

month, day, year = map(int, input().split())

day_semaine = calendar.weekday(year, month, day)

days = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

print(days[day_semaine])

#P2: Time Delta

import os
from datetime import datetime

def time_delta(t1, t2):
    time_format = "%a %d %b %Y %H:%M:%S %z"
    d1 = datetime.strptime(t1, time_format)
    d2 = datetime.strptime(t2, time_format)
    delta_seconds = abs(int((d1 - d2).total_seconds()))
    return delta_seconds

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(str(delta) + '\n')
    fptr.close()


# In[ ]:


# Exceptions

#P1 : Exceptions

n = int(input())

for i in range(n):
    a, b = input().split()
    try:
        a = int(a)
        b = int(b)
        print(a // b)
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)


#P2 : Incorrect Regex

import re 

n= int(input())

for i in range(n):
    pattern = input()
    try:
        re.compile(pattern)
        print("True")
    except re.error:
        print("False")


# In[ ]:


# Built-Ins

#P1: Input()

x, k = map(int, input().split())
polynomial = input()

if eval(polynomial) == k:
    print(True)
else:
    print(False)


#P2: Any or All

n = int(input())
nombres = input().split()
positive = all(int(x) > 0 for x in nombres)
palindrome = any(x == x[::-1] for x in nombres if int(x) > 0)
resultat = positive and palindrome
print(resultat)


# In[ ]:


# Python fonctionnals 

#P1 : 

def fibonacci(n):
    fib = []
    a, b = 0, 1
    for i in range(n):
        fib.append(a)
        a, b = b, a + b
    return fib


# In[ ]:


# Regex and Parsing

#P1 : Detect Floating Point Number

import re

n = int(input())

for k in range(n):
    N = input()
    pattern = r'^[+-]?(\d*\.\d+|\d+\.\d+)$'
    if re.match(pattern, N):
        try:
            float(N)
            print(True)
        except ValueError:
            print(False)
    else : 
        print(False)
    

#P2 : Re.split()

regex_pattern = r'[,.]'

#P3 : Group(), Groups() & Groupdict()

import re

pattern = r'([a-zA-Z0-9])\1+'

N = input()

match = re.search(pattern,N)

if match :
    print(match.group(1))
else:
    print(-1)

#P4 : Re.findall() & Re.finditer()

import re 

N = input()

pattern = r'(?<=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])([aeiouAEIOU]{2,})(?=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])'

matchs = re.findall(pattern, N)

if matchs :
    for i in matchs:
        print(i)
else : 
    print(-1)


#P5 : Re.start() & Re.end()

import re
S = input()
k = input()
pattern = re.compile(f'(?={re.escape(k)})')
matches = list(pattern.finditer(S))

if not matches:
    print((-1,-1))

else : 
    for match in matches:
        print((match.start(), match.end()+len(k)-1))

#P6 : Validating Roman Numerals

regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

#P7 : Validating phone numbers

import re
pattern = r'^[789]\d{9}$'

n = int(input())

for i in range(n):
    num = input().strip()
    if re.match(pattern, num):
        print("YES")
    else: 
        print("NO")

#P8 : Validating and Parsing Email Addresses

import re 
import email.utils

pattern = r'^[a-zA-Z][\w._-]*@[a-zA-Z]+\.[a-zA-Z]{2,4}$'

n = int(input())

for i in range(n):
    a = input().strip()
    name, email = email.utils.parseaddr(a)
    
if re.match(pattern, email):
    print(f"{name} <{email}>")

#P9 : Hex Color Code

#P10  : Validating UID

import re

def is_valid_uid(uid):
    if len(uid) != 10:
        return False
    if not uid.isalnum():
        return False
    if len(re.findall(r'[A-Z]', uid)) < 2:
        return False
    if len(re.findall(r'\d', uid)) < 3:
        return False
    if len(set(uid)) != len(uid):
        return False
    return True

n = int(input().strip())

for i in range(n):
    uid = input().strip()
    
    if is_valid_uid(uid):
        print("Valid")
    else:
        print("Invalid")


# In[ ]:


# XML 

#P1 : Find the score

def get_attr_number(node):
    score = len(node.attrib)
    for child in node:
        score += get_attr_number(child)
    return score


#P2 : Find the maximum depth

maxdepth = 0

def depth(elem, level):
    global maxdepth
    level += 1
    maxdepth = max(maxdepth, level)
    for child in elem:
        depth(child, level)




# In[ ]:


# Closure and Decorations

#P1: 

def person_lister(f):
    def inner(people):
        people.sort(key=lambda person: int(person[2])) 
        return [f(person) for person in people] 
    return inner


# In[ ]:


# Numpy 

#P1: Arrays
import numpy
def arrays(arr):
    a=numpy.array(arr,float)
    return(a[::-1])
    
#P2: Shape and Reshape

import numpy
liste = list(input().split())
my_array = numpy.array(liste,int)

print (numpy.reshape(my_array,(3,3)))

#P3: Transpose and Flatten

import numpy as np
values = list(map(int,input().split()))
n = values[0]
m = values[1]

liste = []


for k in range (n):
    ligne = list(map(int,input().split()))
    liste.append(ligne)
my_array = np.array(liste)
print(np.transpose(my_array))
print(my_array.flatten())

#P4: Concatenate

import numpy as np
values = list(map(int,input().split()))
n = values[0]
m = values[1]
p = values[2]
liste_1 = []
liste_2 = []
for k in range(n):
    ligne = list(map(int,input().split()))
    liste_1.append(ligne)
for k in range(n, n+m):
    ligne = list(map(int,input().split()))
    liste_2.append(ligne)
my_array1 = np.array(liste_1)
my_array2 = np.array(liste_2)
print(np.concatenate((my_array1,my_array2), axis=0))

#P5: Zeros and Ones

import numpy as np
values= list(map(int,input().split()))

print ( np.zeros(values, dtype= np.int ))  
print ( np.ones(values , dtype= np.int) )

#P6: Eye and Identity

import numpy as np 
np.set_printoptions(legacy='1.13')

values = list(map(int,input().split()))
n = values[0]
m = values[1]
print(np.eye(n,m,k=0))


#P7: Array Mathematics

import numpy as np
values = list(map(int,input().split()))
n = values[0]
m = values[1]
liste_A = []
liste_B = []

for k in range(n):
    liste = list(map(int,input().split()))
    liste_A.append(liste)
for k in range (n):
    liste = list(map(int,input().split()))
    liste_B.append(liste)

a = np.array(liste_A)
b = np.array(liste_B)

print (np.add(a, b))
print (np.subtract(a, b))
print (np.multiply(a, b))
print (np.floor_divide(a, b))
print (np.mod(a, b))
print (np.power(a, b))

#P8: Floor, Ceil and Rint

import numpy as np
np.set_printoptions(legacy='1.13')
v = np.array(list(map(float,input().split())))
print(np.floor(v))
print(np.ceil(v))
print(np.rint(v))

#P9: Sum and Prod

import numpy as np
values = list(map(int,input().split()))
n = values[0]
m = values[1]
l = []
for k in range (n):
    ligne = list(map(int,input().split()))
    l.append(ligne)
my_array = np.array(l)
s = np.sum(my_array, axis =0)
print(np.prod(s))

#P10: Min and Max

import numpy as np
values = list(map(int,input().split()))
n = values[0]
m = values[1]
l =[]
for k in range (n):
    ligne = list(map(int,input().split()))
    l.append(ligne)
my_array= np.array(l)
min_f = np.min(my_array,axis=1)
print(np.max(min_f))
    
#P11: Dot and Cross

import numpy as np
N = int(input())
A = np.array([list(map(int, input().split())) for i in range(N)])
B = np.array([list(map(int, input().split())) for i in range(N)])

print(np.dot(A,B))

#P12: Inner and Outer

import numpy as np
A = np.array( list(map(int, input().split() )))
B = np.array( list(map(int, input().split() )))

print(np.inner(A,B))
print(np.outer(A,B))

#P13: Polynomials

import numpy as np
coeff = list(map(float,input().split()))
value = float(input())
print(np.polyval(coeff, value))



# In[ ]:


# PROBLEM 2 :

#P1 : Birthday Cake Candles

def birthdayCakeCandles(candles):
    max_candle_height = max(candles)
    return(candles.count(max_candle_height))
    
#P2 : Kangaroo 

def kangaroo(x1, v1, x2, v2):
    if (x2-x1)*(v1-v2)>0:
        if (x2-x1) % (v1-v2) == 0:
            return 'YES'
    return 'NO'


#P3 : Strange advertising

def viralAdvertising(n):
    share = [5]
    likes =[]
    for k in range (n):
        likes.append(floor(share[k]/2))
        share.append(likes[k] * 3)
    return(sum(likes))

#P4 : Recursive digit sum 


def superDigit(n, k):
    
    nb = sum(map(int,str(n)))*k
    def get_superDigit(x):
        if x< 10:
            return(x)
        else:
            return(get_superDigit(sum(map(int,str(x)))))
    return(get_superDigit(nb))

#P5 : Insertion Sort 1 

def insertionSort1(n, arr):
    a = arr[-1]
    for k in range(2, n + 1):
        if arr[-k] > a:
            arr[-k + 1] = arr[-k]
            print(" ".join(map(str, arr)))
        else:
            arr[-k + 1] = a
            print(" ".join(map(str, arr)))
            break
    else:
        arr[0] = a
        print(" ".join(map(str, arr)))
        

#P6 : Insertion Sort 2 

def insertionSort2(n, arr):
    for i in range(1, n):
        x = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > x:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = x
        print(" ".join(map(str, arr)))


