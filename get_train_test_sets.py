__author__ = 'thomas'

import json

business_ids = []
f1 = open('reviews_1_small.json','w')
f2 = open('reviews_2_small.json','w')
f3 = open('reviews_3_small.json','w')
f4 = open('reviews_4_small.json','w')
f5 = open('reviews_5_small.json','w')
counter = 0
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0

for review in open('reviews_all.json'):
	if counter < 5000:
		stars = json.loads(review)['stars']
		if stars == 1 and count_1 < 1000:
			f1.write(review)
			counter += 1
			count_1 += 1
		elif stars == 2 and count_2 < 1000:
			f2.write(review)
			counter += 1
			count_2 += 1
		elif stars == 3 and count_3 < 1000:
			f3.write(review)
			counter += 1
			count_3 += 1
		elif stars == 4 and count_4 < 1000:
			f4.write(review)
			counter += 1
			count_4 += 1
		elif stars == 5 and count_5 < 1000:
			f5.write(review)
			counter += 1
			count_5 += 1


f1.close()
f2.close()
f3.close()
f4.close()
f5.close()