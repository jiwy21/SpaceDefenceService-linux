

# from Utils.io import item_to_map
#
#
#
# a = [0, 1, 2]
# b = item_to_map(a)
# print(b)

from datetime import datetime

str = '2020-06-18 23:34:07'
str2date = datetime.strptime(str, "%Y-%m-%d %H:%M:%S")
print(str2date)




