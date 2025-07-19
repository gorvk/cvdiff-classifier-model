const tokenizer = (str) =>
  str
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "") // remove special characters
    .split(/\s+/) // split by whitespace
    .filter(Boolean);


export default tokenizer;