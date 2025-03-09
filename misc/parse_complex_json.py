def parse(S):
    mylist = []
    index = -1
    #由于continue,index不能放在循环体最后，进而index<=len(S)-2
    
    begin_index = None
    end_index = None

    while index < len(S)-1:
        index += 1
        sym = S[index] #存储字符
        
        if sym == '[' or sym == '{':

            if len(mylist) == 0:
                # 第一个字符 
                begin_index = index
            mylist.append(sym) #有左括号时压栈

        #注意：分类讨论
        else:
            if sym == ']':
                if len(mylist)>0 and mylist.pop()=='[':
                    continue
                else:
                    return None
            if sym == '}':
                if len(mylist)>0 and mylist.pop()=='{':
                    continue
                else:
                    return None 

        if begin_index != None and len(mylist) == 0:
            end_index = index
            json_data = S[begin_index: end_index]
            return json_data

    return None

if __name__ == "__main__":
    test_txt = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/json_outputs_try1/reprocess_fail_unparsed_value/978.txt"
    with open(test_txt, "rb") as txt_file:
        text = txt_file.read().decode('utf-8', errors='ignore')
    json_data = parse(text)

    if json_data == None:
        text_remove_first_brackets = text.replace('{', '', 1)
        json_data = parse(text_remove_first_brackets)

    print(json_data)