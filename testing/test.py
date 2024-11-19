import dotenv
dotenv.load_dotenv()

from model_wrappers import gemini, llama, gpt

g = llama()
print(g.prompt("./logos/Windows 7.Noise1.png"))
