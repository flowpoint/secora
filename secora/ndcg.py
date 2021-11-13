import math


def NDCG(list=None):

  score = 0
  ideal_score = 0
  ideal_list = sorted(list, reverse=True)
  for i in range(0, len(list)):
    print(i)
    score += list[i]/math.log2(i+2)
    ideal_score += ideal_list[i]/math.log2(i+2)
  score = score/ideal_score
  return score


#list=[2,3,3,1,2]
#print(NDCG(list))
