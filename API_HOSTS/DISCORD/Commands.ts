
import { bold, Client, GatewayIntentBits, REST, Routes } from "discord.js";
import { OpenAIK, DiscordK } from "./tokens";

const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMembers,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.GuildMessageReactions,
        GatewayIntentBits.GuildMessagePolls
    ]
});

const rest = new REST({version: '10'}).setToken(String(DiscordK));


client.on("interactionCreate", async interaction => {})

client.on("messageCreate", async message =>  {
});