import json

# Load the JSON data from file
with open('test.json', 'r') as file:
    data = json.load(file)

# Define the mapping for new solution formats
solution_mapping = {
    "A": "solution1",
    "B": "solution2",
    "C": "solution3",
    "D": "solution4",
    "E": "solution5"
}

# Update the instruction and answer in the JSON data
for item in data:
    # Update instruction
    item["instruction"] = item["instruction"].replace("(A)", "\n\nsolution1:").replace("(B)", "\nsolution2:").replace("(C)", "\nsolution3:").replace("(D)", "\nsolution4:").replace("(E)", "\nsolution5:")
    item["instruction"] += "\n\nAnswer format: solution1/solution2/solution3/solution4/solution5"
    
    # Update answer
    item["answer"] = solution_mapping.get(item["answer"], item["answer"])

# Save the updated JSON data back to file
with open('updated_test.json', 'w') as file:
    json.dump(data, file, indent=4)

print("File updated successfully!")
