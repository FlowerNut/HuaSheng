import time

t1=time.process_time()
print(t1)
for i in range(10000):
    pass

t2=time.process_time()
print(t2)

t3=(t2-t1)*1000

print(t3)
t4 = 60-t3
print(t4)