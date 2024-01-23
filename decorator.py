import datetime
from functools import wraps
import os

def error_handler(func):
    # log.txtが存在しなければ作成
    if not os.path.isfile:
        with open ("log.txt", "w") as log:
            log.write("")
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            now = datetime.datetime.now()
            now = now.strftime("%Y_%m_%d_%H_%M_%S")
            with open ("log.txt", "a") as log:
                log.write(f"The time error occurred: {now}, The function name: {func.__name__}, Error name: ({type(e).__name__}): {e}\n")
            print(f"An error occurred: {e} ({type(e).__name__})")
    return wrapper

"""
# 1回目の実行時にはFalseを、
# 2回目の実行時にはTrueを返すデコレータ
def once(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not wrapper.called:
            wrapper.called = True
            #print("1回目の動作を実行します。")
            wrapper.bool = False
            func(*args, **kwargs)
            return wrapper.bool  # デコレートされた関数の戻り値を返す
        else:
            #print("2回目以降は特定の動作をスキップします.")
            wrapper.bool = True
            func(*args, **kwargs)
            return wrapper.bool
    
    wrapper.called = False
    return wrapper
"""
"""
## Usage of once function
@once
def my_function():
    print(my_function.bool)
    print("関数が実行されました.")

# テスト
result_1 = my_function()  # 1回目の呼び出し
result_2 = my_function()  # 2回目の呼び出し
my_function()

print("1回目の結果:", result_1)  # 1回目の呼び出しでは関数の戻り値が返される
print("2回目の結果:", result_2)  # 2回目の呼び出しではデコレータからの False が返される
"""


"""
# Example usage:

@error_handler
def divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError as e:
        print("ERROR OCCURRED")
        print(e)
        # Optionally handle the exception here if needed
        raise  # Re-raise the exception to be caught by the decorator
    

@error_handler
def greet(name:int):
    print(f"Hello, {name}!")

# Testing the decorated functions

divide_result = divide(10, 2)
print("Result of divide function:", divide_result)


divide_error = divide(10, 0)  # This will trigger an error


greet("Alice")
greet(123)  
"""