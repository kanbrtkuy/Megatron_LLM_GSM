import re

def extract_answers_from_first_file(file_path):
    """
    从第一个格式的文本文件中提取所有"Answer:"后面的数值
    
    参数:
        file_path (str): 文本文件的路径
        
    返回:
        list: 包含所有提取出的数值的列表
    """
    # 创建一个空列表用于存储提取的答案
    answers = []
    
    try:
        # 打开并读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 使用正则表达式匹配"Answer:"后面的数字
        answer_pattern = re.compile(r'Answer:\s*(\d+)')
        
        # 查找所有匹配项
        matches = answer_pattern.finditer(content)
        
        # 将找到的每个数字转换为整数并添加到列表中
        for match in matches:
            number = int(match.group(1))
            answers.append(number)
            
        print(f"成功从第一个文件中提取了{len(answers)}个答案")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
    except Exception as e:
        print(f"处理第一个文件时发生错误: {e}")
    
    return answers

def extract_answers_from_second_file(file_path):
    """
    从第二个格式的文本文件中提取所有":"后面但在"########"前的数字
    
    参数:
        file_path (str): 文本文件的路径
        
    返回:
        list: 包含所有提取出的数值的列表
    """
    # 创建一个空列表用于存储提取的答案
    answers = []
    
    try:
        # 打开并读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 将文本按分隔符"########"分割成问题块
        questions = content.split('########')
        
        # 处理每个问题块
        for i, question in enumerate(questions[:-1]):  # 最后一个通常是空的
            # 查找最后一个冒号及其后面的内容
            last_colon_index = question.rfind(':')
            
            if last_colon_index != -1:
                # 提取冒号后面的内容直到结束
                after_colon = question[last_colon_index + 1:].strip()
                
                # 从提取的内容中解析数字
                number_match = re.search(r'\d+', after_colon)
                
                if number_match:
                    number = int(number_match.group())
                    answers.append(number)
                else:
                    print(f"问题 {i+1} 没有找到数字")
            else:
                print(f"问题 {i+1} 没有找到冒号")
        
        print(f"成功从第二个文件中提取了{len(answers)}个数字")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
    except Exception as e:
        print(f"处理第二个文件时发生错误: {e}")
    
    return answers

def compare_lists(list1, list2):
    """
    比较两个列表，计算相同位置上数值相同的比率
    
    参数:
        list1 (list): 第一个列表
        list2 (list): 第二个列表
        
    返回:
        float: 相同率(百分比)
    """
    # 首先检查两个列表的长度是否相同
    if len(list1) != len(list2):
        print(f"警告: 两个列表长度不同! 第一个列表长度: {len(list1)}, 第二个列表长度: {len(list2)}")
        return None
    
    # 如果列表为空，则无法计算相同率
    if len(list1) == 0:
        print("警告: 列表为空!")
        return 0.0
    
    # 计算相同元素的数量
    same_count = sum(1 for a, b in zip(list1, list2) if a == b)
    
    # 计算相同率
    similarity_rate = (same_count / len(list1)) * 100
    
    # 输出详细的比较信息
    print("两个列表的详细比较:")
    for i, (a, b) in enumerate(zip(list1, list2)):
        if a == b:
            print(f"位置 {i}: {a} == {b} ✓")
        else:
            print(f"位置 {i}: {a} != {b} ✗")
    
    return similarity_rate

def main():
    """
    主函数，处理两个文件并比较结果
    """
    file1_path = input("请输入第一个文件的路径: ")
    file2_path = input("请输入第二个文件的路径: ")
    
    # 提取两个文件中的答案
    list1 = extract_answers_from_first_file(file1_path)
    list2 = extract_answers_from_second_file(file2_path)
    
    # 输出提取的列表
    print("\n提取的答案列表:")
    print(f"第一个文件: {list1}")
    print(f"第二个文件: {list2}")
    
    # 比较两个列表
    similarity_rate = compare_lists(list1, list2)
    
    # 输出比较结果
    if similarity_rate is not None:
        print(f"\n两个列表的相同率: {similarity_rate:.2f}%")

if __name__ == "__main__":
    main()