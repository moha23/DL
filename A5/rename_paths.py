f = open("/Users/momo/Downloads/Assignment3/Q1/Knuckle/groundtruth_old.txt", "r+")

new_file = []
for line in f:
	#only_a = line.split(",")
	#name = only_a[0]
	new_name = line.replace(" ", "_")
	new_file.append(new_name)
	#only_a[0] = new_name
	#reqd = ','.join(only_a)
	#print(only_a)
	print(new_name)

with open("/Users/momo/Downloads/Assignment3/Q1/Knuckle/groundtruth1.txt", "w+") as f:
  for i in new_file:
    f.write(i)
