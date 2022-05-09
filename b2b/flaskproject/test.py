search_phrases: dict[str, str] = {'Location': '', 'Capabilities': '',
                                  'Company Name': ''}
temp=search_phrases.copy()
for k, v in search_phrases.items():
    if not v:
        del temp[k]
if(temp):
    print("还有")
else:
    print("没了")
print(temp)