import re

chat1 = 'codebasics: you ask a lot of questions  1235678912, JOabc@xyz.com'
chat2 = 'codebasics: here it is: (123)-567-8912, abc_A82@xyz.com'
chat3 = 'codebasics: yes, phone: 1235678912 email: abc@xyz.com'

# look for the phone number

pattern = '\d{10}' # construct pattern on regex101.com for example
pattern_one = '\(\d{3}\)-\d{3}-\d{4}'
pattern_two = '\d{10}|\(\d{3}\)-\d{3}-\d{4}' # | means or

matches = re.findall(pattern, chat1)

print(matches)
print(re.findall(pattern, chat2)) # no matches
print(re.findall(pattern, chat3))

print(re.findall(pattern_one, chat2)) #this works!!

print("check for ", pattern_two)

print(re.findall(pattern_two, chat3)) # Good
print(re.findall(pattern_two, chat1))
print(re.findall(pattern_two, chat2))

# look for the email

pattern_email = '[a-zA-Z0-9_]*@[a-zA-Z0-9]*\.com'
# to match any domain not just .com
pattern_email_one = '[a-zA-Z0-9_]*@[a-zA-Z0-9]*\.[a-zA-Z]*' 

print(re.findall(pattern_email, chat2))
print(re.findall(pattern_email, chat1))
print(re.findall(pattern_email, chat3))

chat4 = 'codebasics: you ask a lot of questions  1235678912, JOabc@xyz.org merci'
print(re.findall(pattern_email_one, chat4))

# look for a specific order

chat5 = 'codebasics: Hello I am having an issue with my order # 412889912'
chat6 = 'codebasics: Hello I have a problem with my order 412889912'

pattern_order = 'order[^\d]'
print(re.findall(pattern_order, chat5))