export const checkPlagiarism = async (file1, file2) => {
  const formData = new FormData();
  formData.append('file1', file1);
  formData.append('file2', file2);

  try {
    const response = await fetch(`${process.env.REACT_APP_API_URL}/check`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Backend returned:", errorText);
      throw new Error('Failed to check plagiarism');
    }

    const data = await response.json();
    console.log("Success:", data);
    return data;
  } catch (error) {
    console.error("Fetch error:", error);
    throw error;
  }
};
