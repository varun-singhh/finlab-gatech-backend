import os
import shutil
import json
import re
import glob
from django.http import JsonResponse
from sec_edgar_downloader import Downloader
from langchain.llms import GooglePalm
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def generateReport(request):
    year = request.GET.get("year")
    ticker = request.GET.get("ticker")

    dl = Downloader(ticker, "varun.singh@" + ticker + ".com")
    try:
        dl.get("10-K", ticker, after=str(year) + "-01-01", before=str(int(year) + 1) + "-01-01", download_details=True)
    except Exception as e:
        print(f"Error downloading {ticker} filings for {str(year)}-01-01-{str(int(year) + 1)}-01-01")
        return JsonResponse({'error':{"code":"500","status":"Internal server error","message":"something went wrong, please try again"}},status=500)


    file_path = os.path.join("sec-edgar-filings", ticker, "10-K", "*", "primary-document.html")
    html_files = glob.glob(file_path)

    html_content = ""
    if html_files:
        html_file_path = html_files[0]  # Take the first HTML file found
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        shutil.rmtree("sec-edgar-filings")
    else:
        print(f"no html file found")
        return JsonResponse({'error':{"code":"500","status":"Internal server error","message":"something went wrong, please try again"}},status=500)


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    chunks = text_splitter.create_documents([html_content])


    chunks_prompt="""Summarize the provided financial data, including the Income Statement, Balance Sheet, Cash Flow Statement, and Financial Metrics. The Income Statement reveals a net income of $ReferenceData["incomeStatement"]["netIncome"] billion, with corresponding metrics such as Net Income Per Share, Diluted Earnings Per Share, EBITDA, Operating Income, and Revenue. The Balance Sheet showcases Total Assets, Total Liabilities, and Total Equity, along with details on Cash and Cash Equivalents, Short-Term Investments, and Accounts Receivable Net, all valued in billions of dollars. The Cash Flow Statement outlines Net Cash Provided by Operating Activities, Net Cash Used in Investing Activities, and Net Cash Used in Financing Activities, alongside the beginning and ending balances of Cash and Cash Equivalents. Finally, Financial Metrics include Revenue, Net Income, Earnings Per Share, and Free Cash Flow, all denoted in billions of dollars. Additionally, businessInsights include Revenue Growth, Net Income Growth, Earnings Per Share Growth, and Free Cash Flow Growth are provided as percentages.
    Data:`{text}'
    Summary:
    """
    map_prompt_template=PromptTemplate(input_variables=['text'],template=chunks_prompt)
        

    final_combine_prompt='''Json data for the sumarized data from provided financial data, including the Income Statement, Balance Sheet, Cash Flow Statement, and Financial Metrics. The Income Statement reveals a net income of $ReferenceData["incomeStatement"]["netIncome"] billion, with corresponding metrics such as Net Income Per Share, Diluted Earnings Per Share, EBITDA, Operating Income, and Revenue. The Balance Sheet showcases Total Assets, Total Liabilities, and Total Equity, along with details on Cash and Cash Equivalents, Short-Term Investments, and Accounts Receivable Net, all valued in billions of dollars. The Cash Flow Statement outlines Net Cash Provided by Operating Activities, Net Cash Used in Investing Activities, and Net Cash Used in Financing Activities, alongside the beginning and ending balances of Cash and Cash Equivalents. Finally, Financial Metrics include Revenue, Net Income, Earnings Per Share, and Free Cash Flow, all denoted in billions of dollars. Additionally, businessInsights include  Revenue Growth, Net Income Growth, Earnings Per Share Growth, and Free Cash Flow Growth are provided as percentages.
    Data: `{text}`
    '''
    final_combine_prompt_template=PromptTemplate(input_variables=['text'],template=final_combine_prompt)

    llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.7)

    chunks = chunks[40:42]

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=final_combine_prompt_template,
        verbose=False
    )

    output = summary_chain.run(chunks)

    clean_json = re.sub(r'```json|```|\n', '', output)

    # Replace single quotes with double quotes
    clean_json = clean_json.replace("\'", "\"")

    # Parse JSON
    try:
        data = json.loads(clean_json)
    except Exception as e:
        print(f"Error parsing {ticker} filing data for {str(year)}-01-01-{str(int(year) + 1)}-01-01")
        return JsonResponse({'error':{"code":"500","status":"Internal server error","message":"something went wrong, please try again"}},status=500)

    return JsonResponse({"code":"200","status":"Success",'data': data},status=200)