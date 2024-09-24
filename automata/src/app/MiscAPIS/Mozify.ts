import { SpotifyApi } from "@spotify/web-api-ts-sdk";

import * as _ManageCookies from "../Backend/Cookies";

require("dotenv").config();

import SpotifyJSON from "@Globals/SpotifyAPI.json";

const CLIENT_SECRET = process.env.spotify_clientSecret;
const CLIENT_ID = process.env.spotify_clientID!;

const auth = SpotifyApi.withUserAuthorization(
    CLIENT_ID,
    SpotifyJSON.REDIRECT_URI,
    SpotifyJSON.Web_Visible_Scopes
);

const TOKEN = auth.getAccessToken();

async function getMyFavoriteTracks(limit: number = 20, offset: number = 0) {
    const response = ""
}