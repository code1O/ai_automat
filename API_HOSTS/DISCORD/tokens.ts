/**
 * 
 * Prepare all the project's tokens
 * even for discord and external like openai
 * 
*/

require('dotenv').config();

export const OpenAIK = process.env.OpenAIK
export const DiscordK = process.env.automata_discord_credential

