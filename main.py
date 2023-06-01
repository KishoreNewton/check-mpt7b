from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mosaicml/mpt-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

input_text = """
Give a json output for this resume:

John Smith
123 Main Street
New York, NY 10001
Phone: (555) 123-4567
Email: johnsmith@email.com

Objective:
Results-oriented sales professional with a proven track record of exceeding targets and driving revenue growth. Seeking a challenging sales position in a dynamic organization where I can leverage my skills and experience to achieve sales objectives and contribute to company success.

Education:
Bachelor of Business Administration (BBA)
University of New York, New York, NY
Year of Graduation: 2020

Experience:
Sales Representative
ABC Company, New York, NY
Dates of Employment: March 2020 - Present

Developed and implemented strategic sales plans to target new clients and expand customer base, resulting in a 25% increase in sales revenue.
Conducted market research to identify customer needs and preferences, and utilized findings to tailor sales presentations and product offerings.
Built and maintained strong relationships with key accounts, negotiating contracts and ensuring customer satisfaction.
Trained and mentored junior sales team members, providing guidance on sales techniques and best practices.
Collaborated with cross-functional teams, including marketing and product development, to drive product improvements and identify new market opportunities.

Sales Associate
XYZ Retail Store, New York, NY
Dates of Employment: June 2018 - February 2020

Consistently achieved and surpassed monthly sales targets, earning recognition as the top-performing sales associate for three consecutive quarters.
Provided exceptional customer service, assisting customers in product selection and addressing inquiries to drive customer satisfaction and loyalty.
Maintained an organized and visually appealing sales floor, ensuring product availability and creating an attractive shopping environment.
Collaborated with the store management team to implement promotional campaigns and sales events, resulting in increased foot traffic and revenue.

Skills:

Strong interpersonal and communication skills
Proven ability to build and maintain client relationships
Excellent negotiation and persuasive selling techniques
Proficient in CRM software and sales analytics tools
Results-driven with a focus on achieving and exceeding sales targets
Ability to work effectively in a team-oriented environment

References:
Available upon request
"""
input_tokens = tokenizer.encode(input_text, return_tensors="pt")
output_tokens = model.generate(input_tokens, max_length=5000)
output_text = tokenizer.decode(output_tokens[0])
print(output_tokens)
print(output_text)
