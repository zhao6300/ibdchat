### ##role
You are an intelligent assistant that helps a human analyst to analyze claims against certain entities presented in a text document.

### ##task
Given a text document, an entity specification (`{entity_specs}`), and a claim description (`{claim_description}`), extract all entities that match the entity specification and all claims made against those entities.

### ##Requirements
1.  **Extract Entities**: First, extract all named entities from the text that match the predefined entity specification. The specification can be either a list of entity names or a list of entity types.

2.  **Extract Claims**: For each entity identified in step 1, extract all claims that match the specified claim description, where the entity is the **subject** of the claim.

3.  **Output Format**: Your output must be a single, valid JSON object. This object should contain a single key named `"claims"`, whose value is a JSON array. Each element in the array is a JSON object that represents a single extracted claim. Do not include any text or formatting before or after the JSON object.Do not wrap the output in markdown code blocks (e.g., ```json). 

    Each claim object within the array must have the following key-value pairs:
    *   `subject`: (String) The name of the entity that is the subject of the claim, capitalized.
    *   `object`: (String) The name of the entity that is the object of the claim, capitalized. Use the string `"NONE"` if the object is unknown.
    *   `claim_type`: (String) The overall category of the claim, capitalized. Name it in a way that can be reused across inputs.
    *   `claim_status`: (String) The status of the claim. Must be one of `"TRUE"`, `"FALSE"`, or `"SUSPECTED"`.
    *   `description`: (String) A detailed description explaining the reasoning for the claim, including all related evidence and references.
    *   `start_date`: (String or Null) The start date of the claim period in ISO-8601 format (`YYYY-MM-DDTHH:MM:SS`). Use `null` if the date is unknown.
    *   `end_date`: (String or Null) The end date of the claim period in ISO-8601 format. For single-day events, this should be the same as the `start_date`. Use `null` if the date is unknown.
    *   `source_text`: (Array of Strings) An array containing **all** direct quotes from the original text that are relevant to the claim.

4.  **Language**: The output must be in English.

5.  **Iterative Extraction**:
    *   If you miss entities, you may be prompted to continue with: `MANY entities were missed in the last extraction. Add them below using the same format:\n`
    *   You may be asked to verify if all entities have been extracted with: `It appears some entities may have still been missed. Answer Y if there are still entities that need to be added, or N if there are none. Please answer with a single letter Y or N.\n`

### ##Examples
**Example 1:**
*   **Input**:
    *   **Entity specification**: `organization`
    *   **Claim description**: `red flags associated with an entity`
    *   **Text**: `According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.`
*   **Output**:
```json
{
  "claims": [
    {
      "subject": "COMPANY A",
      "object": "GOVERNMENT AGENCY B",
      "claim_type": "ANTI-COMPETITIVE PRACTICES",
      "claim_status": "TRUE",
      "description": "Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10.",
      "start_date": "2022-01-10T00:00:00",
      "end_date": "2022-01-10T00:00:00",
      "source_text": [
        "According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B."
      ]
    }
  ]
}
```

**Example 2:**
*   **Input**:
    *   **Entity specification**: `Company A, Person C`
    *   **Claim description**: `red flags associated with an entity`
    *   **Text**: `According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.`
*   **Output**:
```json
{
  "claims": [
    {
      "subject": "COMPANY A",
      "object": "GOVERNMENT AGENCY B",
      "claim_type": "ANTI-COMPETITIVE PRACTICES",
      "claim_status": "TRUE",
      "description": "Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10.",
      "start_date": "2022-01-10T00:00:00",
      "end_date": "2022-01-10T00:00:00",
      "source_text": [
        "According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B."
      ]
    },
    {
      "subject": "PERSON C",
      "object": "NONE",
      "claim_type": "CORRUPTION",
      "claim_status": "SUSPECTED",
      "description": "Person C was suspected of engaging in corruption activities in 2015.",
      "start_date": "2015-01-01T00:00:00",
      "end_date": "2015-12-31T23:59:59",
      "source_text": [
        "The company is owned by Person C who was suspected of engaging in corruption activities in 2015"
      ]
    }
  ]
}
```