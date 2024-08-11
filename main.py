#import libraries
import os
import openai
import dotenv
import json
import fitz

# Defining the data for 'Propety' function
def property_function():
    property_data=[
    {
            "name": "extract_attributes",
            "description": "Extract property, Lease details, Demographics mentioned in the text",
            "parameters": {
                "type": "object",
                "properties": {
                    "Property": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Address": {
                                    "type": "string",
                                    "description": "Address of the property"
                                },
                                "RentableArea": {
                                    "type": "number",
                                    "description": "Rentable area of the property"
                                },
                                "NetOperatingIncome": {
                                    "type": "number",
                                    "description": "Net operating income of the property"
                                },
                                "LandArea": {
                                    "type": "number",
                                    "description": "Land area of the property"
                                },
                                "CapRate": {
                                    "type": "number",
                                    "description": "Cap rate of the property"
                                },
                                "YearBuilt": {
                                    "type": "string",
                                    "description": "Year built of the property"
                                },
                                "SalePrice": {
                                    "type": "number",
                                    "description": "Sale price of the property"
                                },
                                "Tenancy": {
                                    "type": "string",
                                    "description": "Tenancy of the property"
                                },
                                "ExpirationDate": {
                                    "type": "string",
                                    "description": "Expiration date of the property"
                                },
                                "TaxKey": {
                                    "type": "string",
                                    "description": "Tax key of the property"
                                },
                                "Taxes2021": {
                                    "type": "number",
                                    "description": "Taxes 2021 of the property"
                                },
                                "Municipality": {
                                    "type": "string",
                                    "description": "Municipality of the property"
                                }                                
                            }, 
                            "required": ["Address", "RentableArea","NetOperatingIncome","LandArea","CapRate","YearBuilt","SalePrice","Tenancy","ExpirationDate","TaxKey","Taxes2021","Municipality"]
                        }
                    }
                },
                "required": ["Property"],
            },
        },
    ]
    return property_data

#Defining the data for 'LeaseDetails' function
def LeaseDetails_function():
    LeaseDetails_data=[
    {
            "name": "extract_attributes",
            "description": "Extract property and Lease details mentioned in the text",
            "parameters": {
                "type": "object",
                "properties": {
                    "LeaseDetails": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "AnnualBaseRent": {
                                    "type": "number",
                                    "description": "Annual Base Rent of the property"
                                },
                                "CommonAreaMaintenance": {
                                    "type": "number",
                                    "description": "Common Area Maintenance of the property"
                                },
                                "Insurance": {
                                    "type": "string",
                                    "description": "Insurance of the property"
                                },
                                "RealEstateTaxes": {
                                    "type": "number",
                                    "description": "Real Estate Tax of the property"
                                },
                                "ExpirationDate": {
                                    "type": "string",
                                    "description": "Expiration Date of the property"
                                },
                                "BaseRentIncrease": {
                                    "type": "number",
                                    "description": "Base Rent Increase of the property"
                                },
                                "Options": {
                                    "type": "string",
                                    "description": "Options of the property"
                                },
                                "Grantor": {
                                    "type": "string",
                                    "description": "Grantor of the property"
                                }
                            },

                            "required": ["AnnualBaseRent", "CommonAreaMaintenance","Insurance","RealEstateTaxes","ExpirationDate","BaseRentIncrease","Options","Grantor"]
                        }
                    }
                },
                "required": ["LeaseDetails"],
            },
        },
    ]
    return LeaseDetails_data

def extract_information(text, functions):
    #call the LLM model with functions and the user's input
    messages=[{"role": "user", "content": text}]
    
    #Call the OpenAI API's chat completions endpoint
    completion = openai.chat.completions.create(
    model="gpt-4", 
    messages=messages,
    functions =functions,
    function_call={"name":"extract_attributes"}
    )
    choice = completion.choices[0]
    extracted_data = choice.message.function_call.arguments
    
    return extracted_data
    
if __name__=='__main__':
    #read pdf file and convert to text
    try:
        doc = fitz.open('./data/nw1.pdf')
    except IOError:
        print ("Could not read file:")

    text_to_analyze =""
    for page in doc:
        text_to_analyze += page.get_text()

     #extract Property attributes from the text
    property_funcData =property_function()
    extracted_data1 =extract_information(text_to_analyze, property_funcData)

    #write into .json file
    try: 
        json_object1 = json.dumps(extracted_data1, indent=4)
        with open("Property.json", "w") as file:
            file.write(json_object1)
    except (json.JSONDecodeError, IndexError):
        pass

    #extract leaseDetails from the text
    LeaseDetails_funcData =LeaseDetails_function()
    extracted_data2 =extract_information(text_to_analyze, LeaseDetails_funcData)

    #write into .json file
    try: 
        json_object2 = json.dumps(extracted_data2, indent=4)
        with open("LeaseDetails.json", "w") as file:
            file.write(json_object2)
    except (json.JSONDecodeError, IndexError):
        pass
    


