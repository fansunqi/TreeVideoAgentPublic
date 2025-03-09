import json

with open("results/egoschema/depth_expansion_res.json", "r") as f:
    depth_res_org = json.load(f)

# sorted_values_sum = 0
# res_num = 0
# for res in depth_res_org:
#     sorted_values_sum += len(res['sorted_values'])
#     res_num += 1
# sorted_values_avernum = sorted_values_sum / res_num
# print(sorted_values_avernum)  # 62.486

for res in depth_res_org:
    res['sorted_values'] = [i for i in range(3, 169, 3)]
with open("results/egoschema/depth_expansion_res_new.json", "w") as f:
    json.dump(depth_res_org, f, indent=4)